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
 * Staged interface for complex double-precision SVD (zgesdd).
 *
 * Splits plasma_zgesdd into five independently callable stages:
 *
 *   Stage 0 - plasma_zgesdd_init:
 *       Validate arguments, populate the context, create PLASMA descriptors
 *       and workspace, allocate all intermediate buffers, and translate the
 *       input matrix A into tile layout (ge2desc).  Contains one-time
 *       initialisation overhead; exclude from Roofline / performance analysis.
 *
 *   Stage 1 - plasma_zgesdd_bidiag_stage1:
 *       Reduce the input matrix A to a banded form (ge2gb) and copy to
 *       LAPACK band layout (gecpy_tile2lapack_band).  Pure computation;
 *       suitable for performance analysis.
 *
 *   Stage 2 - plasma_zgesdd_bidiag_stage2:
 *       Bulge-chase the band to bidiagonal form (gbbrd).
 *
 *   Stage 3 - plasma_zgesdd_dq:
 *       Solve the bidiagonal SVD via divide-and-conquer (bdsdc) and
 *       initialise the singular vector matrices pU and pVT.
 *
 *   Stage 4 - plasma_zgesdd_backtransform:
 *       Back-transform the singular vectors through all accumulated
 *       Householder reflectors (Q2/P2 from bulge chasing, Q1/P1 from
 *       band reduction).
 *
 * Typical usage from C++:
 *
 *   plasma_zgesdd_ctx_t ctx;
 *   plasma_zgesdd_init(jobu, jobvt, m, n, pA, lda, &ctx);
 *   plasma_zgesdd_bidiag_stage1(&ctx);
 *   plasma_zgesdd_bidiag_stage2(&ctx);
 *   plasma_zgesdd_dq(S, pU, ldu, pVT, ldvt, &ctx);
 *   plasma_zgesdd_backtransform(pU, ldu, pVT, ldvt, &ctx);
 *
 **/
#ifndef PLASMA_ZGESDD_STAGES_H
#define PLASMA_ZGESDD_STAGES_H

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *
 * Context structure that carries intermediate SVD state between the five
 * pipeline stages.  The caller allocates the struct (stack or heap); its
 * contents are initialised by plasma_zgesdd_init and released by
 * plasma_zgesdd_backtransform (or plasma_zgesdd_ctx_destroy on early exit).
 *
 ******************************************************************************/
typedef struct {
    /* ---- configuration ---- */
    plasma_enum_t jobu;
    plasma_enum_t jobvt;
    plasma_enum_t uplo;       /* PlasmaUpper if m >= n, else PlasmaLower */
    char          lapack_uplo; /* 'U' or 'L'                              */
    int m, n, minmn, nb;
    int Un;                   /* number of left  singular vectors to form */
    int VTm;                  /* number of right singular vectors to form */

    /* ---- bulge-chasing blocking ---- */
    int vblksiz;  /* blocking size inside bulge chasing          */
    int blkcnt;   /* number of Householder blocks (wantz != 0)   */
    int ldv;      /* leading dimension of V arrays               */
    int ldt;      /* leading dimension of T arrays               */
    int lda_band; /* leading dimension of band storage (3*nb+1)  */
    int wantz;    /* 0 = no vectors, 2 = want vectors            */

    /* ---- PLASMA tile descriptors and workspace ---- */
    plasma_desc_t      A;
    plasma_desc_t      T;
    plasma_workspace_t work;
    plasma_sequence_t  sequence;
    plasma_request_t   request;

    /* ---- intermediate buffers (owned by context) ---- */
    double             *S_bidiag; /* bidiagonal diagonal,   size minmn   */
    double             *E;        /* bidiagonal off-diag,   size minmn   */
    plasma_complex64_t *pA_band;  /* band matrix in LAPACK band layout    */

    /* reflectors accumulated during bulge chasing – left (Q side) */
    plasma_complex64_t *VQ2;
    plasma_complex64_t *tauQ2;
    plasma_complex64_t *TQ2;      /* NULL when wantz == 0                */

    /* reflectors accumulated during bulge chasing – right (P side) */
    plasma_complex64_t *VP2;
    plasma_complex64_t *tauP2;
    plasma_complex64_t *TP2;      /* NULL when wantz == 0                */
} plasma_zgesdd_ctx_t;

/***************************************************************************//**
 *
 * Stage 0: Initialisation (excluded from performance analysis).
 *
 * Validates arguments, populates the context, creates PLASMA descriptors and
 * workspace, allocates all intermediate buffers, and translates the input
 * matrix A into tile layout (plasma_pzge2desc).
 *
 * This stage contains one-time setup overhead and should NOT be included in
 * Roofline or performance counter measurements.
 *
 * @param[in]  jobu
 *     PlasmaAllVec / PlasmaSomeVec / PlasmaNoVec
 *
 * @param[in]  jobvt
 *     PlasmaAllVec / PlasmaSomeVec / PlasmaNoVec  (must equal jobu)
 *
 * @param[in]  m, n
 *     Dimensions of the input matrix.  m >= 0, n >= 0, min(m,n) >= nb.
 *
 * @param[in,out] pA
 *     m-by-n matrix in column-major (LAPACK) layout.
 *     On exit the contents are destroyed (overwritten by tile layout).
 *
 * @param[in]  lda
 *     Leading dimension of pA.  lda >= max(1, m).
 *
 * @param[out] ctx
 *     Caller-allocated context.  Zero-initialised internally on entry.
 *     Must not be freed by the caller; use plasma_zgesdd_ctx_destroy or
 *     let plasma_zgesdd_backtransform release resources on success.
 *
 * @retval PlasmaSuccess on success, negative error code otherwise.
 *
 ******************************************************************************/
int plasma_zgesdd_init(
    plasma_enum_t jobu, plasma_enum_t jobvt,
    int m, int n,
    plasma_complex64_t *pA, int lda,
    plasma_zgesdd_ctx_t *ctx);

/***************************************************************************//**
 *
 * Stage 1: Reduction to banded form (pure computation).
 *
 * Must be called after plasma_zgesdd_init.
 * Performs the tile reduction A → Band (plasma_pzge2gb) and copies the
 * result to LAPACK band layout (plasma_pzgecpy_tile2lapack_band).
 *
 * This stage contains no initialisation overhead and is suitable for
 * Roofline / performance counter measurements.
 *
 * @param[in,out] ctx
 *     Context initialised by plasma_zgesdd_init.
 *
 * @retval PlasmaSuccess on success, negative error code otherwise.
 *
 ******************************************************************************/
int plasma_zgesdd_bidiag_stage1(plasma_zgesdd_ctx_t *ctx);

/***************************************************************************//**
 *
 * Stage 2: Bulge-chasing to bidiagonal form.
 *
 * Must be called after plasma_zgesdd_bidiag_stage1.
 * Runs plasma_pzgbbrd_static on the band matrix stored in ctx.
 * On return:
 *   - ctx->S_bidiag holds the bidiagonal diagonal entries
 *   - ctx->E        holds the bidiagonal off-diagonal entries
 *   - VQ2/tauQ2/TQ2 and VP2/tauP2/TP2 hold the Householder reflectors
 *     needed for the back-transformation in stage 3
 *
 * @param[in,out] ctx
 *     Context initialised by plasma_zgesdd_bidiag_stage1.
 *
 * @retval PlasmaSuccess on success, negative error code otherwise.
 *
 ******************************************************************************/
int plasma_zgesdd_bidiag_stage2(plasma_zgesdd_ctx_t *ctx);

/***************************************************************************//**
 *
 * Stage 3: Divide-and-conquer bidiagonal SVD.
 *
 * Must be called after plasma_zgesdd_bidiag_stage2.
 * Performs:
 *   1. LAPACKE_dbdsdc – bidiagonal D&C SVD
 *   2. Initialises pU and pVT with the bidiagonal singular vectors
 *      (real→complex copy for COMPLEX builds)
 *
 * On success ctx remains valid and must be passed to
 * plasma_zgesdd_backtransform (or freed via plasma_zgesdd_ctx_destroy).
 * On error ctx is freed internally.
 *
 * @param[out] S
 *     Array of at least min(m,n) doubles.
 *     On exit holds the singular values in descending order.
 *
 * @param[out] pU
 *     Left singular-vector matrix.  Ignored when jobu == PlasmaNoVec.
 *
 * @param[in]  ldu     Leading dimension of pU.
 *
 * @param[out] pVT
 *     Right singular-vector matrix (V^H).  Ignored when jobvt == PlasmaNoVec.
 *
 * @param[in]  ldvt    Leading dimension of pVT.
 *
 * @param[in,out] ctx
 *     Context from stage 2.  On success kept alive for stage 4.
 *     On error freed internally; do not use after a failing call.
 *
 * @retval PlasmaSuccess on success, negative error code otherwise.
 *
 ******************************************************************************/
int plasma_zgesdd_dq(
    double *S,
    plasma_complex64_t *pU,  int ldu,
    plasma_complex64_t *pVT, int ldvt,
    plasma_zgesdd_ctx_t *ctx);

/***************************************************************************//**
 *
 * Stage 4: Back-transform singular vectors.
 *
 * Must be called after plasma_zgesdd_dq.
 * Performs:
 *   1. plasma_pzlarft_blgtrd + plasma_pzunmqr_blgtrd – apply Q2 / P2
 *      (bulge-chasing reflectors)
 *   2. plasma_pzunmqr / plasma_pzunmlq               – apply Q1 / P1
 *      (band-reduction reflectors)
 *
 * A no-op (other than freeing ctx) when jobu == PlasmaNoVec and
 * jobvt == PlasmaNoVec.
 *
 * Frees all internal buffers and descriptors held in ctx on exit
 * (equivalent to calling plasma_zgesdd_ctx_destroy).
 *
 * @param[in,out] pU
 *     Left singular-vector matrix, as initialised by plasma_zgesdd_dq.
 *     On exit contains the fully back-transformed left singular vectors.
 *     Ignored when jobu == PlasmaNoVec.
 *
 * @param[in]  ldu     Leading dimension of pU.
 *
 * @param[in,out] pVT
 *     Right singular-vector matrix (V^H), as initialised by plasma_zgesdd_dq.
 *     On exit contains the fully back-transformed right singular vectors.
 *     Ignored when jobvt == PlasmaNoVec.
 *
 * @param[in]  ldvt    Leading dimension of pVT.
 *
 * @param[in,out] ctx
 *     Context from stage 3.  Freed internally; do not use after this call.
 *
 * @retval PlasmaSuccess on success, negative error code otherwise.
 *
 ******************************************************************************/
int plasma_zgesdd_backtransform(
    plasma_complex64_t *pU,  int ldu,
    plasma_complex64_t *pVT, int ldvt,
    plasma_zgesdd_ctx_t *ctx);

/***************************************************************************//**
 *
 * Free all resources owned by ctx.
 *
 * Safe to call at any point after plasma_zgesdd_bidiag_stage1 has returned
 * (even if stage 2 or 3 have not been called), and safe to call on a
 * zero-initialised struct.
 *
 ******************************************************************************/
void plasma_zgesdd_ctx_destroy(plasma_zgesdd_ctx_t *ctx);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PLASMA_ZGESDD_STAGES_H */
