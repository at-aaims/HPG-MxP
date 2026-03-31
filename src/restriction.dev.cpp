/** ************************************************************************
 * @file restriction.cpp
 * @copyright (c) 2019-2021 Advanced Micro Devices, Inc.,
 *            (c) 2025 Oak Ridge National Laboratory.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * ************************************************************************ */

#include "restriction.hpp"

#include "DataTypes.hpp"
#include "hpgmp.hpp"
#include "ell_matrix.hpp"

#ifdef HPGMP_WITH_GINKGO
#include "GinkgoMatrix.hpp"
#endif

#include "kernel_helpers.hpp.inc"

#define LAUNCH_FUSED_RESTRICT_SPMV(blocksize, width)                                           \
    {                                                                                          \
        dim3 blocks((A.mgData->rc->local_length() - 1) / blocksize + 1);                       \
        dim3 threads(blocksize);                                                               \
                                                                                               \
        kernel_fused_restrict_spmv<blocksize, width><<<blocks, threads, 0, stream_interior>>>( \
            A.mgData->rc->local_length(),                                                      \
            A.mgData->d_f2cOperator,                                                           \
            rf.d_values(),                                                                     \
            A.localNumberOfRows,                                                               \
            A.localNumberOfColumns,                                                            \
            mat->get_ld_indices(), mat->get_ld_values(),                                       \
            mat->get_col_idxs(),                                                               \
            mat->get_values(),                                                                 \
            xf.d_values(),                                                                     \
            A.mgData->rc->d_values(),                                                          \
            A.perm,                                                                            \
            A.Ac->perm);                                                                       \
    }

template<unsigned int BLOCKSIZE, typename scalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_restrict(local_int_t size,
                                    const local_int_t* __restrict__ f2cOperator,
                                    const scalar* __restrict__ fine,
                                    const scalar* __restrict__ data,
                                    scalar* __restrict__ coarse,
                                    const local_int_t* __restrict__ perm_fine,
                                    const local_int_t* __restrict__ perm_coarse)
{
    const local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx_coarse >= size) {
        return;
    }
    const local_int_t idx_fine      = perm_fine[f2cOperator[idx_coarse]];
    coarse[perm_coarse[idx_coarse]] = fine[idx_fine] - data[idx_fine];
}

template<unsigned int BLOCKSIZE, unsigned int WIDTH, typename mat_scalar, typename vec_scalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_fused_restrict_spmv(const local_int_t size,
                                               const local_int_t* f2cOperator,
                                               const vec_scalar* r_fine,
                                               const local_int_t m, const local_int_t n,
                                               const int ldi, const int ldv,
                                               const local_int_t* __restrict__ ell_col_ind,
                                               const mat_scalar* ell_val,
                                               const vec_scalar* xf,
                                               vec_scalar* r_coarse,
                                               const local_int_t* __restrict__ perm_fine,
                                               const local_int_t* __restrict__ perm_coarse)
{
    const local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx_coarse >= size) {
        return;
    }
    const local_int_t idx_fine        = __ldcg(f2cOperator + idx_coarse);
    const local_int_t idx_perm_fine   = __ldcg(perm_fine + idx_fine);
    const local_int_t idx_perm_coarse = __ldcg(perm_coarse + idx_coarse);

    vec_scalar sum = __ldcg(r_fine + idx_perm_fine);

#pragma unroll
    for (local_int_t p = 0; p < WIDTH; ++p)
    {
        const local_int_t col = __ldcg(ell_col_ind + idx_perm_fine + p * ldi);

        if (col >= 0 && col < m) {
            sum = fma(-static_cast<vec_scalar>(
                          __ldcg(ell_val + idx_perm_fine + p * ldv)),
                      xf[col], sum);
        }
    }
    __stcg(r_coarse + idx_perm_coarse, sum);
}

template<unsigned int BLOCKSIZE, typename mat_scalar, typename vec_scalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_fused_restrict_spmv_halo(const local_int_t m,
                                                    const local_int_t n,
                                                    const local_int_t* __restrict__ c2fOperator,
                                                    const local_int_t halo_width,
                                                    const int ldi, const int ldv,
                                                    const local_int_t* __restrict__ halo_row_ind,
                                                    const local_int_t* __restrict__ halo_col_ind,
                                                    const mat_scalar* __restrict__ halo_val,
                                                    const vec_scalar* __restrict__ xf,
                                                    vec_scalar* __restrict__ coarse,
                                                    const local_int_t* __restrict__ perm_coarse)
{
    const local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (row >= m) {
        return;
    }

    const local_int_t idx_coarse = c2fOperator[halo_row_ind[row]];
    // Check if halo row contributes to coarse vector, else discard it
    if (idx_coarse == -1) {
        return;
    }

    vec_scalar sum = 0.0;

    for (local_int_t p = 0; p < halo_width; ++p)
    {
        local_int_t col = halo_col_ind[p * ldi + row];

        if (col >= 0 && col < n) {
            sum = fma(halo_val[p * ldv + row], xf[col], sum);
        }
    }
    coarse[perm_coarse[idx_coarse]] -= sum;
}

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<typename mscalar, typename vscalar>
int restriction(const SparseMatrix<mscalar>& A, const Vector<vscalar>& rf)
{
    auto stream_interior = A.dctx->get_compute_stream();
    dim3 blocks((A.mgData->rc->local_length() - 1) / 128 + 1);
    dim3 threads(128);

    kernel_restrict<128><<<blocks, threads, 0, stream_interior>>>(
        A.mgData->rc->local_length(),
        A.mgData->d_f2cOperator,
        rf.d_values(),
        A.mgData->Axf->d_values(),
        A.mgData->rc->d_values(),
        A.perm,
        A.Ac->perm);

    return 0;
}

template int restriction(const SparseMatrix<float>& A, const Vector<float>& rf);
template int restriction(const SparseMatrix<double>& A, const Vector<double>& rf);

template<typename mscalar, typename vscalar>
int fused_spmv_restriction(const SparseMatrix<mscalar>& A, const Vector<vscalar>& rf,
                           const Vector<vscalar>& xf)
{
#ifdef HPGMP_WITH_GINKGO
    std::shared_ptr<const GinkgoMatrix<mscalar, mscalar>> mat =
        dynamic_cast<GinkgoOptData<mscalar, mscalar>*>(A.optimizationData)->mat;
#else
    std::shared_ptr<const ELLMatrix<mscalar, mscalar>> mat =
        dynamic_cast<EllOptData<mscalar, mscalar>*>(A.optimizationData)->mat;
#endif
    auto stream_interior = mat->get_device_context()->get_compute_stream();
#ifndef HPCG_NO_MPI
    if (A.geom->size > 1)
    {
        xf.update_halos_pack_send_buffer(mat.get());
    }
#endif

    if (mat->get_ell_width() == 27) {
        LAUNCH_FUSED_RESTRICT_SPMV(1024, 27);
    }

#ifndef HPCG_NO_MPI
    if (A.geom->size > 1)
    {
        xf.update_halos_send_receive(mat.get());
        xf.update_halos_finalize(mat.get());

        dim3 blocks((mat->get_num_halo_rows() - 1) / 128 + 1);
        dim3 threads(128);

        kernel_fused_restrict_spmv_halo<128><<<blocks,
                                               threads,
                                               0,
                                               stream_interior>>>(
            mat->get_num_halo_rows(), mat->get_local_num_cols(),
            A.mgData->d_c2fOperator,
            mat->get_ell_width(), mat->get_halo_ld_indices(), mat->get_halo_ld_values(),
            mat->get_halo_row_indices(), mat->get_halo_col_idxs(),
            mat->get_halo_values(),
            xf.d_values(), A.mgData->rc->d_values(),
            A.Ac->perm);
    }
#endif
    mat->get_device_context()->synchronize_compute_stream();
    mat->get_device_context()->synchronize_halo_stream();
    return 0;
}

template int fused_spmv_restriction(const SparseMatrix<double>& A, const Vector<double>& rf,
                                    const Vector<double>& xf);

template int fused_spmv_restriction(const SparseMatrix<float>& A, const Vector<float>& rf,
                                    const Vector<float>& xf);
