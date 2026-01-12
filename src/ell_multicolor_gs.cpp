
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/* ************************************************************************
 * Modifications (c) 2019-2021 Advanced Micro Devices, Inc.
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

#include "ell_multicolor_gs.hpp"

#include "hpgmp.hpp"
#include "DataTypes.hpp"

#include "kernel_helpers.hpp.inc"

#define LAUNCH_SYMGS_SWEEP(blocksize, width)                                  \
    {                                                                         \
        dim3 blocks((A->get_independent_set_sizes()[i] - 1) / blocksize + 1); \
        dim3 threads(blocksize);                                              \
                                                                              \
        kernel_symgs_sweep<blocksize, width><<<blocks,                        \
                                               threads,                       \
                                               0,                             \
                                               stream_interior>>>(            \
            A->get_local_num_rows(),                                          \
            A->get_local_num_cols(),                                          \
            A->get_independent_set_sizes()[i],                                \
            A->get_independent_set_offsets()[i],                              \
            A->get_ld_indices(), A->get_ld_values(),                          \
            A->get_col_idxs(),                                                \
            A->get_values(),                                                  \
            A->get_inverse_diagonal(),                                        \
            r->d_values(),                                                    \
            x->d_values());                                                   \
    }

#define LAUNCH_SYMGS_INTERIOR(blocksize, width)                               \
    {                                                                         \
        dim3 blocks((A->get_independent_set_sizes()[0] - 1) / blocksize + 1); \
        dim3 threads(blocksize);                                              \
                                                                              \
        kernel_symgs_interior<blocksize, width><<<blocks,                     \
                                                  threads,                    \
                                                  0,                          \
                                                  stream_interior>>>(         \
            A->get_local_num_rows(),                                          \
            A->get_independent_set_sizes()[0],                                \
            A->get_ld_indices(), A->get_ld_values(),                          \
            A->get_col_idxs(),                                                \
            A->get_values(),                                                  \
            A->get_inverse_diagonal(),                                        \
            r->d_values(),                                                    \
            x->d_values());                                                   \
    }

#define LAUNCH_SYMGS_HALO(blocksize, width)                        \
    {                                                              \
        dim3 blocks((A->get_num_halo_rows() - 1) / blocksize + 1); \
        dim3 threads(blocksize);                                   \
                                                                   \
        kernel_symgs_halo<blocksize, width><<<blocks,              \
                                              threads,             \
                                              0,                   \
                                              stream_interior>>>(  \
            A->get_num_halo_rows(),                                \
            A->get_local_num_cols(),                               \
            A->get_independent_set_sizes()[0],                     \
            A->get_halo_ld_indices(), A->get_halo_ld_values(),     \
            A->get_halo_row_indices(),                             \
            A->get_halo_col_idxs(),                                \
            A->get_halo_values(),                                  \
            A->get_inverse_diagonal(),                             \
            A->get_reordering_permutation(),                       \
            r->d_values(),                                         \
            x->d_values());                                        \
    }

template<unsigned int BLOCKSIZE, unsigned int WIDTH,
         typename mscalar, typename diag_scalar, typename vscalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_symgs_sweep(const local_int_t m,
                                       const local_int_t n,
                                       const local_int_t block_nrow,
                                       const local_int_t offset,
                                       const int ldi, const int ldv,
                                       const local_int_t* ell_col_ind,
                                       const mscalar* ell_val,
                                       const diag_scalar* inv_diag,
                                       const vscalar* x,
                                       vscalar* y)
{
    const local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (gid >= block_nrow) {
        return;
    }

    const local_int_t row = gid + offset;

    vscalar sum = __ldcg(x + row);

#pragma unroll
    for (local_int_t p = 0; p < WIDTH; ++p)
    {
        const local_int_t col = __ldcg(ell_col_ind + row + ldi * p);

        if (col >= 0 && col < n && col != row) {
            sum = fma(-static_cast<vscalar>(__ldcg(ell_val + row + ldv * p)),
                      y[col], sum);
        }
    }
    __stcg(y + row,
           sum * static_cast<vscalar>(__ldcg(inv_diag + row)));
}

template<unsigned int BLOCKSIZE, unsigned int WIDTH,
         typename mscalar, typename diag_scalar, typename vscalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_symgs_interior(const local_int_t m,
                                          const local_int_t block_nrow,
                                          const int ldi, const int ldv,
                                          const local_int_t* ell_col_ind,
                                          const mscalar* ell_val,
                                          const diag_scalar* inv_diag,
                                          const vscalar* x,
                                          vscalar* y)
{
    const local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (row >= block_nrow) {
        return;
    }

    vscalar sum = __ldcg(x + row);

#pragma unroll
    for (local_int_t p = 0; p < WIDTH; ++p)
    {
        const local_int_t col = __ldcg(ell_col_ind + row + p * ldi);

        if (col >= 0 && col < m && col != row) {
            sum = fma(-static_cast<vscalar>(__ldcg(ell_val + row + p * ldv)),
                      __ldg(y + col), sum);
        }
    }
    __stcg(y + row,
           sum * static_cast<vscalar>(__ldcg(inv_diag + row)));
}

template<unsigned int BLOCKSIZE, unsigned int WIDTH,
         typename mscalar, typename diag_scalar, typename vscalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_symgs_halo(const local_int_t m,
                                      const local_int_t n,
                                      const local_int_t block_nrow,
                                      const int ldi, const int ldv,
                                      const local_int_t* halo_row_ind,
                                      const local_int_t* halo_col_ind,
                                      const mscalar* halo_val,
                                      const diag_scalar* inv_diag,
                                      const local_int_t* perm,
                                      const vscalar* x,
                                      vscalar* y)
{
    const local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (row >= m) {
        return;
    }

    const local_int_t halo_idx = __ldcg(halo_row_ind + row);
    const local_int_t perm_idx = perm[halo_idx];
    if (perm_idx >= block_nrow) {
        return;
    }

    vscalar sum = 0.0;

#pragma unroll
    for (local_int_t p = 0; p < WIDTH; ++p)
    {
        const local_int_t col = __ldcg(halo_col_ind + row + p * ldi);

        if (col >= 0 && col < n) {
            sum = fma(-static_cast<vscalar>(__ldcg(halo_val + row + p * ldv)),
                      y[col], sum);
        }
    }
    y[perm_idx] = fma(sum, static_cast<vscalar>(inv_diag[halo_idx]), y[perm_idx]);
}

template<unsigned int BLOCKSIZE, typename scalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_pointwise_mult(const local_int_t size,
                                          const scalar* __restrict__ x,
                                          const scalar* __restrict__ y,
                                          scalar* __restrict__ out)
{
    const local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (gid >= size) {
        return;
    }
    out[gid] = x[gid] * y[gid];
}

template<unsigned int BLOCKSIZE, typename mscalar, typename vscalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_forward_sweep_0(const local_int_t m,
                                           const local_int_t block_nrow,
                                           const local_int_t offset,
                                           const int ldi, const int ldv,
                                           const local_int_t* ell_col_ind,
                                           const mscalar* ell_val,
                                           const local_int_t* diag_idx,
                                           const vscalar* x,
                                           vscalar* y)
{
    const local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (gid >= block_nrow) {
        return;
    }

    const local_int_t row   = gid + offset;
    const local_int_t idiag = __ldcg(diag_idx + row);

    vscalar sum = __ldcg(x + row);

    for (local_int_t p = 0; p < idiag; ++p)
    {
        const local_int_t col = __ldcg(ell_col_ind + row + p * ldi);

        // Every entry above offset is zero
        if (col >= 0 && col < offset) {
            sum = fma(-static_cast<vscalar>(__ldcg(ell_val + row + p * ldv)),
                      y[col], sum);
        }
    }
    sum = sum / static_cast<vscalar>(__ldcg(ell_val + row + idiag * ldv));
    __stcg(y + row, sum);
}

template<unsigned int BLOCKSIZE, typename mscalar, typename vscalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_backward_sweep_0(const local_int_t m,
                                            const local_int_t block_nrow,
                                            const local_int_t offset,
                                            const local_int_t ell_width,
                                            const int ldi, const int ldv,
                                            const local_int_t* ell_col_ind,
                                            const mscalar* ell_val,
                                            const local_int_t* diag_idx,
                                            vscalar* x)
{
    const local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (gid >= block_nrow) {
        return;
    }

    const local_int_t row   = gid + offset;
    const local_int_t idiag = __ldcg(diag_idx + row);

    const mscalar diag_val = __ldcg(ell_val + row + idiag * ldv);

    // Scale result with diagonal entry
    vscalar sum = x[row] * static_cast<vscalar>(diag_val);

    for (local_int_t p = idiag + 1; p < ell_width; ++p)
    {
        const local_int_t col = __ldcg(ell_col_ind + row + p * ldi);

        // Every entry below offset should not be taken into account
        if (col >= offset && col < m) {
            sum = fma(-static_cast<vscalar>(__ldcg(ell_val + row + p * ldv)),
                      x[col], sum);
        }
    }
    sum /= static_cast<vscalar>(diag_val);
    __stcg(x + row, sum);
}

template<typename mscalar, typename vscalar>
int ell_multicolor_gs(const bool symmetric, const ELLMatrix<mscalar>* const A,
                      const Vector<vscalar>* const r, Vector<vscalar>* const x)
{
    assert(x->local_length() == A->get_local_num_cols());
    auto dctx            = A->get_device_context();
    auto stream_interior = dctx->get_compute_stream();

    local_int_t i = 0;

#ifndef HPCG_NO_MPI
    if (A->get_geometry()->size > 1)
    {
        x->update_halos_pack_send_buffer(A);

        if (A->get_ell_width() == 27) {
            LAUNCH_SYMGS_INTERIOR(1024, 27);
        }

        x->update_halos_send_receive(A);
        x->update_halos_finalize(A);

        if (A->get_ell_width() == 27) {
            LAUNCH_SYMGS_HALO(256, 27);
        }

        ++i;
    }
#endif

    // Solve L
    for (; i < A->get_num_independent_sets(); ++i)
    {
        if (A->get_ell_width() == 27) {
            LAUNCH_SYMGS_SWEEP(1024, 27);
        }
    }

    if (symmetric) {
        throw std::runtime_error("symmetric not yet tested!");
        // Solve U
        //for(i = A.ublocks; i >= 0; --i)
        //{
        //    if(A->get_ell_width() == 27) LAUNCH_SYMGS_SWEEP(1024, 27);
        //}
    }

    dctx->synchronize_compute_stream();
    dctx->synchronize_halo_stream();
    return 0;
}

template<typename mscalar, typename vscalar>
int ell_multicolor_gs_zero_initial(const bool symmetric, const ELLMatrix<mscalar>* const A,
                                   const Vector<vscalar>* const r, Vector<vscalar>* const x)
{
    assert(x->local_length() == A->get_local_num_cols());
    auto dctx            = A->get_device_context();
    auto stream_interior = dctx->get_compute_stream();

    // Solve L
    kernel_pointwise_mult<256><<<(A->get_independent_set_sizes()[0] - 1) / 256 + 1,
                                 256,
                                 0,
                                 stream_interior>>>(
        A->get_independent_set_sizes()[0], r->d_values(), A->get_inverse_diagonal(),
        x->d_values());

    for (local_int_t i = 1; i < A->get_num_independent_sets(); ++i)
    {
        kernel_forward_sweep_0<1024><<<(A->get_independent_set_sizes()[i] - 1) / 1024 + 1,
                                       1024,
                                       0,
                                       stream_interior>>>(
            A->get_local_num_rows(),
            A->get_independent_set_sizes()[i], A->get_independent_set_offsets()[i],
            A->get_ld_indices(), A->get_ld_values(),
            A->get_col_idxs(), A->get_values(),
            A->get_diagonal_indices(), r->d_values(), x->d_values());
    }

    if (symmetric) {
        throw std::runtime_error("symmetric not yet tested!");
        // Solve U
        //for(local_int_t i = A.ublocks; i >= 0; --i)
        //{
        //    kernel_backward_sweep_0<1024><<<(A->get_independent_set_sizes()[i] - 1) / 1024 + 1,
        //                                    1024,
        //                                    0,
        //                                    stream_interior>>>(
        //        A.get_local_num_rows(),
        //        A.get_independent_set_sizes()[i], A->get_independent_set_offsets()[i],
        //        A.get_ell_width(),
        //        A->get_ld_indices(), A->get_ld_values(),
        //        A.ell_col_ind, A.ell_val,
        //        A.diag_idx, x.d_values);
        //}
    }

    dctx->synchronize_compute_stream();

    return 0;
}

template int ell_multicolor_gs(
    bool sym, const ELLMatrix<double>* const A, const Vector<double>* const r,
    Vector<double>* const x);

template int ell_multicolor_gs(
    bool sym, const ELLMatrix<float>* const A, const Vector<float>* const r,
    Vector<float>* const x);

template int ell_multicolor_gs_zero_initial(
    bool sym, const ELLMatrix<double>* const A,
    const Vector<double>* const r, Vector<double>* const x);

template int ell_multicolor_gs_zero_initial(
    bool sym, const ELLMatrix<float>* const A,
    const Vector<float>* const r, Vector<float>* const x);
