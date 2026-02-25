/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 * Staged implementation of the complex double-precision SVD.
 * Splits plasma_omp_zgesdd into three separately callable stages so that
 * external code can interleave other work (or insert instrumentation) between
 * the major phases of the computation:
 *
 *   Stage 1 (bidiag_stage1) – reduction to banded form
 *   Stage 2 (bidiag_stage2) – bulge-chasing band → bidiagonal
 *   Stage 3 (dq)            – bidiagonal D&C SVD + back-transform
 *
 **/

#include "plasma_zgesdd_stages.h"

#include "plasma.h"
#include "core_lapack.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "bulge.h"

#include <omp.h>
#include <string.h>
#include <stdlib.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @ingroup plasma_gesdd
 *
 * plasma_zgesdd_ctx_destroy - free all resources owned by a stage context.
 *
 * Safe to call on a zero-initialised struct and at any point after
 * plasma_zgesdd_bidiag_stage1 returns.
 *
 ******************************************************************************/
void plasma_zgesdd_ctx_destroy(plasma_zgesdd_ctx_t *ctx)
{
    free(ctx->S_bidiag); ctx->S_bidiag = NULL;
    free(ctx->E);        ctx->E        = NULL;
    free(ctx->pA_band);  ctx->pA_band  = NULL;
    free(ctx->VQ2);      ctx->VQ2      = NULL;
    free(ctx->tauQ2);    ctx->tauQ2    = NULL;
    free(ctx->TQ2);      ctx->TQ2      = NULL;
    free(ctx->VP2);      ctx->VP2      = NULL;
    free(ctx->tauP2);    ctx->tauP2    = NULL;
    free(ctx->TP2);      ctx->TP2      = NULL;

    plasma_workspace_destroy(&ctx->work);
    plasma_desc_destroy(&ctx->A);
    plasma_desc_destroy(&ctx->T);
}

/***************************************************************************//**
 *
 * @ingroup plasma_gesdd
 *
 * plasma_zgesdd_bidiag_stage1 - reduce A to banded form.
 *
 * Translates pA into tile layout, creates all PLASMA descriptors and
 * workspace, allocates intermediate buffers, and performs:
 *
 *   plasma_pzge2gb               (tile reduction to band)
 *   plasma_pzgecpy_tile2lapack_band  (copy tile band → LAPACK band layout)
 *
 * On return ctx holds everything needed by stage 2.
 *
 ******************************************************************************/
int plasma_zgesdd_bidiag_stage1(
    plasma_enum_t jobu, plasma_enum_t jobvt,
    int m, int n,
    plasma_complex64_t *pA, int lda,
    plasma_zgesdd_ctx_t *ctx)
{
    /* Zero-initialise so ctx_destroy is safe from any point. */
    memset(ctx, 0, sizeof(*ctx));

    /* ---- PLASMA context ---- */
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    /* ---- argument checks (mirrors plasma_zgesdd) ---- */
    int minmn = imin(m, n);
    if (jobu != PlasmaNoVec && jobu != PlasmaAllVec && jobu != PlasmaSomeVec) {
        plasma_error("illegal value of jobu");
        return -1;
    }
    if (jobvt != PlasmaNoVec && jobvt != PlasmaAllVec && jobvt != PlasmaSomeVec) {
        plasma_error("illegal value of jobvt");
        return -2;
    }
    if (jobvt != jobu) {
        plasma_error("in this version: jobu should be equal jobvt");
        return -2;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -4;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -6;
    }
    if (minmn == 0)
        return PlasmaSuccess;

    int ib = plasma->ib;
    int nb = plasma->nb;

    if (minmn < nb) {
        plasma_error("nb < imin(m, n) not supported");
        return -12;
    }

    /* ---- populate context configuration ---- */
    ctx->jobu       = jobu;
    ctx->jobvt      = jobvt;
    ctx->uplo       = (m >= n) ? PlasmaUpper : PlasmaLower;
    ctx->lapack_uplo = (m >= n) ? 'U' : 'L';
    ctx->m          = m;
    ctx->n          = n;
    ctx->minmn      = minmn;
    ctx->nb         = nb;
    ctx->Un         = (jobu  == PlasmaAllVec) ? m : minmn;
    ctx->VTm        = (jobvt == PlasmaAllVec) ? n : minmn;
    ctx->lda_band   = 3*nb + 1;

    ctx->vblksiz = nb / 4;          /* equivalent to ib */
    ctx->ldt     = ctx->vblksiz;
    ctx->wantz   = (jobu == PlasmaNoVec && jobvt == PlasmaNoVec) ? 0 : 2;

    /* ---- create tile matrix A ---- */
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &ctx->A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    /* ---- prepare descriptor T ---- */
    retval = plasma_descT_create(ctx->A, ib, PlasmaFlatHouseholder, &ctx->T);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_descT_create() failed");
        plasma_zgesdd_ctx_destroy(ctx);
        return retval;
    }

    /* ---- allocate workspace ---- */
    size_t lwork = (size_t)ib*nb + 4*(size_t)nb*nb;
    retval = plasma_workspace_create(&ctx->work, lwork, PlasmaComplexDouble);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_workspace_create() failed");
        plasma_zgesdd_ctx_destroy(ctx);
        return retval;
    }

    /* ---- init sequence and request ---- */
    plasma_sequence_init(&ctx->sequence);
    plasma_request_init(&ctx->request);

    /* ---- translate pA to tile layout ---- */
    #pragma omp parallel
    #pragma omp master
    {
        plasma_pzge2desc(pA, lda, ctx->A, &ctx->sequence, &ctx->request);
    }

    /* ---- band storage:
     *   pA_band looks like:
     *        __________________________________
     *  NB   |               zero               |
     *        ----------------------------------
     *  NB+1 |               band A             |
     *        ----------------------------------
     *  NB   |_______________zero_______________|
     *
     *  The gecpy call writes starting at &pA_band[nb].
     * ---- */
    ctx->pA_band = (plasma_complex64_t*)
        malloc((size_t)ctx->lda_band * minmn * sizeof(plasma_complex64_t));
    if (ctx->pA_band == NULL) {
        plasma_error("malloc(pA_band) failed");
        plasma_zgesdd_ctx_destroy(ctx);
        return PlasmaErrorOutOfMemory;
    }
    memset(ctx->pA_band, 0,
           (size_t)ctx->lda_band * minmn * sizeof(plasma_complex64_t));

    /* ---- bidiagonal diagonal / off-diagonal buffers ---- */
    ctx->S_bidiag = (double*) malloc(minmn * sizeof(double));
    ctx->E        = (double*) malloc(minmn * sizeof(double));
    if (ctx->S_bidiag == NULL || ctx->E == NULL) {
        plasma_error("malloc(S_bidiag or E) failed");
        plasma_zgesdd_ctx_destroy(ctx);
        return PlasmaErrorOutOfMemory;
    }

    /* ---- bulge-chasing reflector buffers ---- */
    if (ctx->wantz) {
        int blkcnt, ldv;
        findVTsiz(minmn, nb, ctx->vblksiz, &blkcnt, &ldv);
        ctx->blkcnt = blkcnt;
        ctx->ldv    = ldv;

        ctx->tauQ2 = (plasma_complex64_t*)
            malloc((size_t)blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        ctx->VQ2   = (plasma_complex64_t*)
            malloc((size_t)ldv * blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        ctx->TQ2   = (plasma_complex64_t*)
            malloc((size_t)ctx->ldt * blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));

        ctx->tauP2 = (plasma_complex64_t*)
            malloc((size_t)blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        ctx->VP2   = (plasma_complex64_t*)
            malloc((size_t)ldv * blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        ctx->TP2   = (plasma_complex64_t*)
            malloc((size_t)ctx->ldt * blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));

        if (!ctx->tauQ2 || !ctx->VQ2 || !ctx->TQ2 ||
            !ctx->tauP2 || !ctx->VP2 || !ctx->TP2) {
            plasma_error("malloc of bulge-chasing vector buffers failed");
            plasma_zgesdd_ctx_destroy(ctx);
            return PlasmaErrorOutOfMemory;
        }

        memset(ctx->tauQ2, 0,
               (size_t)    blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        memset(ctx->VQ2,   0,
               (size_t)ldv*blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        memset(ctx->TQ2,   0,
               (size_t)ctx->ldt*blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));

        memset(ctx->tauP2, 0,
               (size_t)    blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        memset(ctx->VP2,   0,
               (size_t)ldv*blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
        memset(ctx->TP2,   0,
               (size_t)ctx->ldt*blkcnt * ctx->vblksiz * sizeof(plasma_complex64_t));
    }
    else {
        /* wantz == 0: only small dummy buffers needed for gbbrd */
        ctx->tauQ2 = (plasma_complex64_t*)
            malloc(2 * minmn * sizeof(plasma_complex64_t));
        ctx->VQ2   = (plasma_complex64_t*)
            malloc(2 * minmn * sizeof(plasma_complex64_t));
        ctx->tauP2 = (plasma_complex64_t*)
            malloc(2 * minmn * sizeof(plasma_complex64_t));
        ctx->VP2   = (plasma_complex64_t*)
            malloc(2 * minmn * sizeof(plasma_complex64_t));
        /* TQ2 and TP2 remain NULL */

        if (!ctx->tauQ2 || !ctx->VQ2 || !ctx->tauP2 || !ctx->VP2) {
            plasma_error("malloc of bulge-chasing dummy buffers failed");
            plasma_zgesdd_ctx_destroy(ctx);
            return PlasmaErrorOutOfMemory;
        }

        memset(ctx->tauQ2, 0, 2 * minmn * sizeof(plasma_complex64_t));
        memset(ctx->VQ2,   0, 2 * minmn * sizeof(plasma_complex64_t));
        memset(ctx->tauP2, 0, 2 * minmn * sizeof(plasma_complex64_t));
        memset(ctx->VP2,   0, 2 * minmn * sizeof(plasma_complex64_t));
    }

    /* ---- reduction to band ---- */
    #pragma omp parallel
    #pragma omp master
    {
        plasma_pzge2gb(ctx->A, ctx->T, ctx->work,
                       &ctx->sequence, &ctx->request);

        plasma_pzgecpy_tile2lapack_band(ctx->uplo, ctx->A,
                                        &ctx->pA_band[nb], ctx->lda_band,
                                        &ctx->sequence, &ctx->request);
    }

    return ctx->sequence.status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gesdd
 *
 * plasma_zgesdd_bidiag_stage2 - bulge-chase band matrix to bidiagonal.
 *
 * Must be called after plasma_zgesdd_bidiag_stage1.
 *
 * Calls plasma_pzgbbrd_static on the band matrix stored in ctx.
 * On return ctx->S_bidiag and ctx->E hold the bidiagonal diagonal and
 * off-diagonal, and the reflector arrays (VQ2/VP2/…) have been filled.
 *
 ******************************************************************************/
int plasma_zgesdd_bidiag_stage2(plasma_zgesdd_ctx_t *ctx)
{
    plasma_pzgbbrd_static(ctx->uplo, ctx->minmn, ctx->nb, ctx->vblksiz,
                          ctx->pA_band, ctx->lda_band,
                          ctx->VQ2, ctx->tauQ2,
                          ctx->VP2, ctx->tauP2,
                          ctx->S_bidiag, ctx->E,
                          ctx->wantz,
                          ctx->work,
                          &ctx->sequence, &ctx->request);

    return ctx->sequence.status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gesdd
 *
 * plasma_zgesdd_dq - bidiagonal D&C SVD and back-transformation.
 *
 * Must be called after plasma_zgesdd_bidiag_stage2.
 *
 * Performs:
 *   1. LAPACKE_dbdsdc       – bidiagonal divide-and-conquer SVD
 *   2. larft_blgtrd + unmqr_blgtrd – apply Q2 / P2 (bulge reflectors)
 *   3. unmqr / unmlq               – apply Q1 / P1 (band reflectors)
 *
 * Frees all resources held in ctx before returning (equivalent to
 * calling plasma_zgesdd_ctx_destroy).
 *
 ******************************************************************************/
int plasma_zgesdd_dq(
    double *S,
    plasma_complex64_t *pU,  int ldu,
    plasma_complex64_t *pVT, int ldvt,
    plasma_zgesdd_ctx_t *ctx)
{
    int retval = PlasmaSuccess;

    int m      = ctx->m;
    int n      = ctx->n;
    int minmn  = ctx->minmn;
    int nb     = ctx->nb;
    int Un     = ctx->Un;
    int VTm    = ctx->VTm;
    int wantz  = ctx->wantz;
    int vblksiz = ctx->vblksiz;

    plasma_enum_t jobu  = ctx->jobu;
    plasma_enum_t jobvt = ctx->jobvt;

    /* ---- bidiagonal D&C SVD ---- */
    double rdummy[1];
    int    idummy[1];
    int lapack_info;

    if (jobu == PlasmaNoVec && jobvt == PlasmaNoVec) {
        lapack_info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, ctx->lapack_uplo,
                                     'N', minmn,
                                     ctx->S_bidiag, ctx->E,
                                     rdummy, ldu,
                                     rdummy, ldvt,
                                     rdummy, idummy);
        if (lapack_info != 0) {
            plasma_error("dbdsdc() failed");
            retval = PlasmaErrorIllegalValue;
            goto cleanup;
        }
        /* Copy singular values to caller's buffer. */
        memcpy(S, ctx->S_bidiag, minmn * sizeof(double));
    }
    else {
        /* For job = PlasmaAllVec:
         *   U0  = [ Uhat  0 ],  VT0 = [ VThat  0 ]
         *         [  0    I ]          [   0    I ]
         * For job = PlasmaSomeVec:
         *   U0  = [ Uhat ],     VT0 = [ VThat  0 ]
         *         [  0   ]
         */

        /* Initialise pU and pVT to zero. */
        memset(pU,  0, (size_t)ldu  * Un * sizeof(plasma_complex64_t));
        memset(pVT, 0, (size_t)ldvt * n  * sizeof(plasma_complex64_t));

        /* pU(n+1:m, n+1:m) = I  when m > n */
        if (jobu == PlasmaAllVec) {
            for (int i = n; i < m; i++)
                pU[i + (size_t)ldu*i] = 1.0;
        }
        /* pVT(m+1:n, m+1:n) = I  when n > m */
        if (jobvt == PlasmaAllVec) {
            for (int i = m; i < n; i++)
                pVT[i + (size_t)ldvt*i] = 1.0;
        }

#if defined COMPLEX
        /* bdsdc works on real arithmetic; allocate temporary real matrices. */
        double *RU  = (double*) malloc((size_t)minmn * minmn * sizeof(double));
        double *RVT = (double*) malloc((size_t)minmn * minmn * sizeof(double));
        if (RU == NULL || RVT == NULL) {
            plasma_error("malloc RU or RVT failed");
            free(RU);
            free(RVT);
            retval = PlasmaErrorOutOfMemory;
            goto cleanup;
        }

        lapack_info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, ctx->lapack_uplo, 'I',
                                     minmn, ctx->S_bidiag, ctx->E,
                                     RU, minmn, RVT, minmn,
                                     rdummy, idummy);

        /* Copy real RU → complex pU  and  real RVT → complex pVT. */
        for (int j = 0; j < minmn; j++)
            for (int i = 0; i < minmn; i++)
                pU[i + (size_t)ldu*j] = RU[i + (size_t)minmn*j];

        for (int j = 0; j < minmn; j++)
            for (int i = 0; i < minmn; i++)
                pVT[i + (size_t)ldvt*j] = RVT[i + (size_t)minmn*j];

        free(RU);
        free(RVT);
#else
        lapack_info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, ctx->lapack_uplo, 'I',
                                     minmn, ctx->S_bidiag, ctx->E,
                                     pU, ldu, pVT, ldvt,
                                     rdummy, idummy);
#endif
        if (lapack_info != 0) {
            plasma_error("dbdsdc() failed");
            retval = PlasmaErrorIllegalValue;
            goto cleanup;
        }

        /* Copy singular values to caller's buffer. */
        memcpy(S, ctx->S_bidiag, minmn * sizeof(double));

        /* ================================================
         * Back-transform U = Q1 Q2 U0
         * ================================================ */
        if (jobu == PlasmaAllVec || jobu == PlasmaSomeVec) {
            /* Step 1: compute T2 for Q2 */
            #pragma omp parallel
            {
                plasma_pzlarft_blgtrd(minmn, nb, vblksiz,
                                      ctx->VQ2, ctx->TQ2, ctx->tauQ2,
                                      &ctx->sequence, &ctx->request);
            }

            /* Step 2: apply Q2 (from bulge chasing) to U */
            #pragma omp parallel
            {
                plasma_pzunmqr_blgtrd(PlasmaLeft, PlasmaNoTrans,
                                      minmn, nb, minmn, vblksiz, wantz,
                                      ctx->VQ2, ctx->TQ2, ctx->tauQ2,
                                      pU, ldu,
                                      ctx->work,
                                      &ctx->sequence, &ctx->request);
            }

            /* Step 3: apply Q1 (from band reduction) to U */
            plasma_desc_t U;
            plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                       m, Un, 0, 0, m, Un, &U);

            #pragma omp parallel
            #pragma omp master
            {
                /* Translate U to tile layout. */
                plasma_pzge2desc(pU, ldu, U, &ctx->sequence, &ctx->request);

                if (m < n) {
                    plasma_pzunmqr(
                        PlasmaLeft, PlasmaNoTrans,
                        plasma_desc_view(ctx->A,
                                         ctx->A.mb, 0,
                                         ctx->A.m - ctx->A.mb,
                                         ctx->A.n - ctx->A.nb),
                        plasma_desc_view(ctx->T,
                                         ctx->T.mb, 0,
                                         ctx->T.m - ctx->T.mb,
                                         ctx->T.n - ctx->T.nb),
                        plasma_desc_view(U, U.mb, 0, U.m - U.mb, U.n),
                        ctx->work, &ctx->sequence, &ctx->request);
                }
                else {
                    plasma_pzunmqr(PlasmaLeft, PlasmaNoTrans,
                                   ctx->A, ctx->T, U,
                                   ctx->work, &ctx->sequence, &ctx->request);
                }

                /* Translate U back to LAPACK layout. */
                plasma_pzdesc2ge(U, pU, ldu, &ctx->sequence, &ctx->request);
            }

            plasma_desc_destroy(&U);
        }

        /* ================================================
         * Back-transform VT = V0^H P2^H P1^H
         * ================================================ */
        if (jobvt == PlasmaAllVec || jobvt == PlasmaSomeVec) {
            /* Step 1: compute T2 for P2 */
            #pragma omp parallel
            {
                plasma_pzlarft_blgtrd(minmn, nb, vblksiz,
                                      ctx->VP2, ctx->TP2, ctx->tauP2,
                                      &ctx->sequence, &ctx->request);
            }

            /* Step 2: apply P2 (from bulge chasing) to VT */
            #pragma omp parallel
            {
                plasma_pzunmqr_blgtrd(PlasmaRight, PlasmaConjTrans,
                                      minmn, nb, minmn, vblksiz, wantz,
                                      ctx->VP2, ctx->TP2, ctx->tauP2,
                                      pVT, ldvt,
                                      ctx->work,
                                      &ctx->sequence, &ctx->request);
            }

            /* Step 3: apply P1 (from band reduction) to VT */
            plasma_desc_t VT;
            plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                       VTm, n, 0, 0, VTm, n, &VT);

            #pragma omp parallel
            #pragma omp master
            {
                /* Translate VT to tile layout. */
                plasma_pzge2desc(pVT, ldvt, VT, &ctx->sequence, &ctx->request);

                if (m < n) {
                    plasma_pzunmlq(PlasmaRight, PlasmaNoTrans,
                                   ctx->A, ctx->T, VT,
                                   ctx->work, &ctx->sequence, &ctx->request);
                }
                else {
                    plasma_pzunmlq(
                        PlasmaRight, PlasmaNoTrans,
                        plasma_desc_view(ctx->A,
                                         0, ctx->A.nb,
                                         ctx->A.m - ctx->A.mb,
                                         ctx->A.n - ctx->A.nb),
                        plasma_desc_view(ctx->T,
                                         0, ctx->T.nb,
                                         ctx->T.m - ctx->T.mb,
                                         ctx->T.n - ctx->T.nb),
                        plasma_desc_view(VT,
                                         0, VT.nb,
                                         VT.m,
                                         VT.n - VT.nb),
                        ctx->work, &ctx->sequence, &ctx->request);
                }

                /* Translate VT back to LAPACK layout. */
                plasma_pzdesc2ge(VT, pVT, ldvt, &ctx->sequence, &ctx->request);
            }

            plasma_desc_destroy(&VT);
        }
    }

    retval = ctx->sequence.status;

cleanup:
    plasma_zgesdd_ctx_destroy(ctx);
    return retval;
}
