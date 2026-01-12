/*!
 * @file ell_matrix.cpp
 *
 * @copyright (c) 2019-2021 Advanced Micro Devices, Inc.,
 *            modifications (c) 2025 Oak Ridge National Laboratory.
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
 * ************************************************************************
 * */

#include "ell_matrix.hpp"

#include <iostream>

#ifdef HPGMP_WITH_HIP
#include <rocprim/rocprim.hpp>
#elif HPGMP_WITH_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#else
#include <execution>
#include <algorithm>
#endif

#include "Vector.hpp"
#include "exceptions.hpp"

#include "kernel_helpers.hpp.inc"

#include "Profiling.hpp"

template<typename hiscalar, typename loscalar>
ELLMatrix<hiscalar, loscalar>::ELLMatrix(const SparseMatrix<hiscalar>& A)
    : DistMatrixBase(A), ldv_{((local_nrows_ - 1) / pad_mult_v + 1) * pad_mult_v}, ldi_{((local_nrows_ - 1) / pad_mult_i + 1) * pad_mult_i}, ell_width_{27}, col_idxs_{static_cast<local_int_t*>(dctx_->device_alloc(ell_width_ * ldi_ * sizeof(local_int_t)))}, values_{static_cast<hiscalar*>(dctx_->device_alloc(ell_width_ * ldv_ * sizeof(hiscalar)))}
{
    int rank = 0;
#ifndef HPGMP_NO_MPI
    MPI_Comm_rank(comm_, &rank);
#endif
    convert_from_csr(A);
}

template<typename hiscalar, typename loscalar>
ELLMatrix<hiscalar, loscalar>::~ELLMatrix()
{
    dctx_->device_free(col_idxs_);
    dctx_->device_free(values_);
    dctx_->device_free(halo_col_idxs_);
    dctx_->device_free(halo_values_);
    dctx_->device_free(diag_idxs_);
    dctx_->device_free(inv_diag_);
}

#if defined HPGMP_WITH_HIP || defined HPGMP_WITH_CUDA

#define LAUNCH_TO_ELL_COL(blocksizex, blocksizey)                       \
    {                                                                   \
        dim3 blocks((A.localNumberOfRows - 1) / blocksizey + 1);        \
        dim3 threads(blocksizex, blocksizey);                           \
                                                                        \
        kernel_to_ell_col<blocksizex, blocksizey><<<blocks, threads>>>( \
            local_nrows_,                                               \
            ell_width_,                                                 \
            A.d_mtxIndL,                                                \
            ldi_,                                                       \
            col_idxs_,                                                  \
            d_n_halo_rows,                                              \
            halo_row_ind_);                                             \
    }

#define LAUNCH_TO_ELL_VAL(blocksizex, blocksizey)                       \
    {                                                                   \
        dim3 blocks((A.localNumberOfRows - 1) / blocksizey + 1);        \
        dim3 threads(blocksizex, blocksizey);                           \
                                                                        \
        kernel_to_ell_val<blocksizex, blocksizey><<<blocks, threads>>>( \
            local_nrows_,                                               \
            max_nnz_per_row,                                            \
            ldv_,                                                       \
            A.d_matrixValues,                                           \
            values_);                                                   \
    }

template<unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX* BLOCKSIZEY)
    __global__ void kernel_to_ell_col(const local_int_t m,
                                      const local_int_t nonzerosPerRow,
                                      const local_int_t* __restrict__ mtxIndL,
                                      const int ldi,
                                      local_int_t* __restrict__ ell_col_ind,
                                      local_int_t* __restrict__ n_halo_rows,
                                      local_int_t* __restrict__ halo_row_ind)
{
    local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

#ifndef HPGMP_NO_MPI
    __shared__ bool sdata[BLOCKSIZEY];
    sdata[threadIdx.y] = false;

    __syncthreads();
#endif

    if (row >= m)
    {
        return;
    }

    const local_int_t col                = __ldg(mtxIndL + row * nonzerosPerRow + threadIdx.x);
    ell_col_ind[threadIdx.x * ldi + row] = col;

#ifndef HPGMP_NO_MPI
    if (col >= m)
    {
        sdata[threadIdx.y] = true;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        if (sdata[threadIdx.y] == true)
        {
            halo_row_ind[atomicAdd(n_halo_rows, 1)] = row;
        }
    }
#endif
}

template<unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY, typename hiscalar>
__launch_bounds__(BLOCKSIZEX* BLOCKSIZEY)
    __global__ void kernel_to_ell_val(const local_int_t m,
                                      const local_int_t nnz_per_row,
                                      const int ld_v,
                                      const hiscalar* __restrict__ matrixValues,
                                      hiscalar* __restrict__ ell_val)
{
    const local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

    if (row >= m) {
        return;
    }

    const local_int_t idx = threadIdx.x * ld_v + row;
    ell_val[idx]          = matrixValues[row * nnz_per_row + threadIdx.x];
}

template<unsigned int BLOCKSIZE, typename hiscalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_to_halo(const local_int_t n_halo_rows,
                                   const local_int_t m,
                                   const local_int_t n,
                                   const local_int_t ell_width,
                                   const int ld_i,
                                   const int ld_v,
                                   const int halo_ld_i,
                                   const int halo_ld_v,
                                   const local_int_t* __restrict__ ell_col_ind,
                                   const hiscalar* __restrict__ ell_val,
                                   const local_int_t* __restrict__ halo_row_ind,
                                   local_int_t* __restrict__ halo_col_ind,
                                   hiscalar* __restrict__ halo_val)
{
    const local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if (gid >= n_halo_rows) {
        return;
    }

    const local_int_t row = halo_row_ind[gid];

    int q = 0;
    for (int p = 0; p < ell_width; ++p)
    {
        //const local_int_t ell_idx = p * m + row;
        const local_int_t col = ell_col_ind[p * ld_i + row];

        if (col >= m && col < n)
        {
            //const local_int_t halo_idx = q++ * n_halo_rows + gid;

            halo_col_ind[q * halo_ld_i + gid] = col;
            halo_val[q * halo_ld_v + gid]     = ell_val[p * ld_v + row];
            q++;
        }
    }

    for (; q < ell_width; ++q)
    {
        //const local_int_t idx = q * n_halo_rows + gid;
        halo_col_ind[q * halo_ld_i + gid] = -1;
    }
}
#endif

local_int_t ref_compute_num_halo_rows(const local_int_t m,
                                      const local_int_t nonzerosPerRow,
                                      const local_int_t* __restrict__ mtxIndL)
{
    local_int_t nhalo = 0;
    for (int i = 0; i < m; i++) {
        bool isRowHalo = false;
        for (int j = 0; j < nonzerosPerRow; j++) {
            const auto col = mtxIndL[i * nonzerosPerRow + j];
            if (col >= m) {
                isRowHalo = true;
            }
        }
        if (isRowHalo) {
            nhalo++;
        }
    }
    return nhalo;
}

template<typename hiscalar, typename loscalar>
void ELLMatrix<hiscalar, loscalar>::convert_from_csr(const SparseMatrix<hiscalar>& A)
{
    const int max_nnz_per_row = ell_width_;

    // Determine blocksize
    unsigned int blocksize = 1024 / ell_width_;

    // Compute next power of two
    blocksize |= blocksize >> 1;
    blocksize |= blocksize >> 2;
    blocksize |= blocksize >> 4;
    blocksize |= blocksize >> 8;
    blocksize |= blocksize >> 16;
    ++blocksize;

    // Shift right until we obtain a valid blocksize
    while (blocksize * ell_width_ > 1024)
    {
        blocksize >>= 1;
    }

    if (blocksize == 32) LAUNCH_TO_ELL_VAL(27, 32)
    else if (blocksize == 16) LAUNCH_TO_ELL_VAL(27, 16)
    else if (blocksize == 8) LAUNCH_TO_ELL_VAL(27, 8)
    else LAUNCH_TO_ELL_VAL(27, 4)

    // Convert mtxIndL into ELL column indices
    local_int_t* d_n_halo_rows = reinterpret_cast<local_int_t*>(dctx_->get_device_workspace());

#ifndef HPGMP_NO_MPI

    dctx_->device_memset(d_n_halo_rows, 0, sizeof(local_int_t));
#endif

    if (blocksize == 32) LAUNCH_TO_ELL_COL(27, 32)
    else if (blocksize == 16) LAUNCH_TO_ELL_COL(27, 16)
    else if (blocksize == 8) LAUNCH_TO_ELL_COL(27, 8)
    else LAUNCH_TO_ELL_COL(27, 4)

#ifndef HPGMP_NO_MPI
    dctx_->copy_device_to_host_sync(&n_halo_rows_, d_n_halo_rows, sizeof(local_int_t));
    assert(n_halo_rows_ <= A.totalToBeSent);

    const local_int_t ref_n_halos = ref_compute_num_halo_rows(
        local_nrows_, ell_width_, A.mtxIndL[0]);
    assert(n_halo_rows_ == ref_n_halos);

    halo_ldv_ = ((n_halo_rows_ - 1) / pad_mult_v + 1) * pad_mult_v;
    halo_ldi_ = ((n_halo_rows_ - 1) / pad_mult_i + 1) * pad_mult_i;

    halo_col_idxs_ = static_cast<local_int_t*>(
        dctx_->device_alloc(halo_ldi_ * ell_width_ * sizeof(local_int_t)));
    halo_values_ = static_cast<hiscalar*>(
        dctx_->device_alloc(halo_ldv_ * ell_width_ * sizeof(hiscalar)));

#ifdef HPGMP_WITH_HIP
    size_t prim_size = 0;
    auto herr        = rocprim::radix_sort_keys(nullptr,
                                                prim_size,
                                                halo_row_ind_,
                                                halo_row_ind_,
                                                n_halo_rows_);
    if (herr != hipSuccess) {
        throw DeviceAPIError("rocprim radix_sort_keys storage estimation");
    }
    auto prim_buffer = dctx_->device_alloc(prim_size);
    herr             = rocprim::radix_sort_keys(prim_buffer,
                                                prim_size,
                                                halo_row_ind_,
                                                halo_row_ind_, // TODO inplace!
                                                n_halo_rows_);
    if (herr != hipSuccess) {
        throw DeviceAPIError("rocprim radix_sort_keys");
    }
    dctx_->device_free(prim_buffer);
#elif HPGMP_WITH_CUDA
    auto d_halo_row_ind = thrust::device_ptr<local_int_t>(halo_row_ind_);
    thrust::stable_sort(d_halo_row_ind, d_halo_row_ind + n_halo_rows_);
#else
    std::stable_sort(std::execution::par_unseq, halo_row_ind_, halo_row_ind_ + n_halo_rows_);
#endif

    kernel_to_halo<128><<<(n_halo_rows_ - 1) / 128 + 1, 128>>>(
        n_halo_rows_, local_nrows_, local_ncols_, ell_width_,
        ldi_, ldv_, halo_ldi_, halo_ldv_,
        col_idxs_, values_,
        halo_row_ind_, halo_col_idxs_, halo_values_);
#endif
}

template<int BLOCKSIZE, typename scalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_permute_ell_rows(const local_int_t m,
                                            const local_int_t p,
                                            const int ldi,
                                            const int ldv,
                                            const local_int_t* __restrict__ tmp_cols,
                                            const scalar* __restrict__ tmp_vals,
                                            const local_int_t* __restrict__ perm,
                                            local_int_t* __restrict__ ell_col_ind,
                                            scalar* __restrict__ ell_val)
{
    const local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (row >= m) {
        return;
    }

    //const local_int_t idx = p * m + perm[row];
    const local_int_t col = tmp_cols[row];

    ell_col_ind[p * ldi + perm[row]] = col;
    ell_val[p * ldv + perm[row]]     = tmp_vals[row];
}

template<typename hiscalar, typename loscalar>
void ELLMatrix<hiscalar, loscalar>::permute_rows(const local_int_t* const perm)
{
    const local_int_t m = local_nrows_;

    // Temporary structures for row permutation
    auto tmp_cols = reinterpret_cast<local_int_t*>(dctx_->device_alloc(m * sizeof(local_int_t)));
    auto tmp_vals = reinterpret_cast<hiscalar*>(dctx_->device_alloc(m * sizeof(hiscalar)));

    // Permute ELL rows
    for (local_int_t p = 0; p < ell_width_; ++p)
    {
        dctx_->copy_device_to_device_sync(tmp_cols, col_idxs_ + p * ldi_, m * sizeof(local_int_t));
        dctx_->copy_device_to_device_sync(tmp_vals, values_ + p * ldv_, m * sizeof(hiscalar));

        kernel_permute_ell_rows<1024><<<(m - 1) / 1024 + 1, 1024>>>(
            m, p, ldi_, ldv_,
            tmp_cols, tmp_vals,
            perm,
            col_idxs_, values_);
    }

    dctx_->device_free(tmp_cols);
    dctx_->device_free(tmp_vals);
}

template<unsigned int BLOCKSIZE, typename mscalar, typename diag_scalar>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_extract_diag_index(const local_int_t m,
                                              const local_int_t ell_width,
                                              const int ldi, const int ldv,
                                              const local_int_t* __restrict__ ell_col_ind,
                                              const mscalar* __restrict__ ell_val,
                                              local_int_t* __restrict__ diag_idx,
                                              diag_scalar* __restrict__ inv_diag)
{
    const local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (row >= m) {
        return;
    }

    for (local_int_t p = 0; p < ell_width; ++p)
    {
        const local_int_t col = ell_col_ind[p * ldi + row];
        if (col == row) {
            diag_idx[row] = p;
            inv_diag[row] = 1.0 / static_cast<diag_scalar>(ell_val[p * ldv + row]);
            break;
        }
    }
}

template<typename hiscalar, typename loscalar>
void ELLMatrix<hiscalar, loscalar>::extract_diagonal()
{
    // Allocate memory to extract diagonal entries
    diag_idxs_ = static_cast<local_int_t*>(
        dctx_->device_alloc(sizeof(local_int_t) * local_nrows_));
    inv_diag_ = static_cast<hiscalar*>(dctx_->device_alloc(sizeof(double) * local_nrows_));

    // Extract diagonal entries
    kernel_extract_diag_index<1024><<<(local_nrows_ - 1) / 1024 + 1, 1024>>>(
        local_nrows_, ell_width_, ldi_, ldv_,
        col_idxs_, values_, diag_idxs_, inv_diag_);
}

template class ELLMatrix<double, double>;
template class ELLMatrix<float, float>;

template<typename mscalar, typename vscalar>
__global__ void simple_ell_spmv(const local_int_t nrows, const int ldv, const int ldi,
                                const local_int_t width,
                                const local_int_t* const col_idxs, const mscalar* const mvalues,
                                const vscalar* const xvals, vscalar* const __restrict__ yvals)
{
    const local_int_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= nrows) {
        return;
    }
    vscalar partial = 0;
    for (int j = 0; j < width; j++) {
        //const auto col = col_idxs[row_idx + j*ldi];
        const local_int_t col = __ldcg(col_idxs + row_idx + j * ldi);
        if (col < 0 || col >= nrows) {
            // skip over halo dependencies; these are taken care of in the halo kernel.
            continue;
        }
        //partial += static_cast<vscalar>(mvalues[row_idx + j*ldv])
        //    * static_cast<vscalar>(xvals[col]);
        partial = fma(static_cast<vscalar>(__ldcg(mvalues + row_idx + j * ldv)),
                      xvals[col], partial);
    }
    //yvals[row_idx] = partial;
    __stcg(yvals + row_idx, partial);
}

template<typename mscalar, typename vscalar>
__global__ void simple_ell_spmv_halo(const local_int_t nrows, const int ldv, const int ldi,
                                     const local_int_t width, const local_int_t* const halo_row_ind,
                                     const local_int_t* const col_idxs, const mscalar* const mvalues,
                                     const local_int_t* const perm,
                                     const vscalar* const xvals, vscalar* const __restrict__ yvals)
{
    const local_int_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= nrows) {
        return;
    }
    vscalar partial = 0;
    for (int j = 0; j < width; j++) {
        const auto col = col_idxs[row_idx + j * ldi];
        if (col < 0) {
            continue;
        }
        partial += static_cast<vscalar>(mvalues[row_idx + j * ldv]) * static_cast<vscalar>(xvals[col]);
    }
    yvals[perm[halo_row_ind[row_idx]]] += partial;
}

template<typename hiscalar, typename vscalar>
void ell_interior_spmv(const ELLMatrix<hiscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y)
{
    constexpr int block_size = 1024;
    int nblocks              = (mat->get_local_num_rows() - 1) / block_size + 1;
    auto stream              = mat->get_device_context()->get_compute_stream();
    simple_ell_spmv<<<nblocks, block_size, 0, stream>>>(mat->get_local_num_rows(),
                                                        mat->get_ld_values(), mat->get_ld_indices(), mat->get_ell_width(),
                                                        mat->get_col_idxs(), mat->get_values(), x->d_values(), y->d_values());
}

template void ell_interior_spmv(const ELLMatrix<double>* mat, const Vector<double>* x, Vector<double>* y);
template void ell_interior_spmv(const ELLMatrix<float>* mat, const Vector<float>* x, Vector<float>* y);
template void ell_interior_spmv(const ELLMatrix<float>* mat, const Vector<double>* x, Vector<double>* y);

template<typename hiscalar, typename vscalar>
void ell_halo_spmv(const ELLMatrix<hiscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y)
{
    constexpr int block_size = 1024;
    const int nblocks        = (mat->get_num_halo_rows() - 1) / block_size + 1;
    // We use the compute stream since the halo spmv *adds* to the output vector y after the interior spmv.
    auto stream = mat->get_device_context()->get_compute_stream();
    simple_ell_spmv_halo<<<nblocks, block_size, 0, stream>>>(mat->get_num_halo_rows(),
                                                             mat->get_halo_ld_values(), mat->get_halo_ld_indices(), mat->get_ell_width(),
                                                             mat->get_halo_row_indices(), mat->get_halo_col_idxs(), mat->get_halo_values(),
                                                             mat->get_reordering_permutation(), x->d_values(), y->d_values());
}

template void ell_halo_spmv(const ELLMatrix<double>* mat, const Vector<double>* x, Vector<double>* y);
template void ell_halo_spmv(const ELLMatrix<float>* mat, const Vector<float>* x, Vector<float>* y);
template void ell_halo_spmv(const ELLMatrix<float>* mat, const Vector<double>* x, Vector<double>* y);

template<typename hiscalar, typename vscalar>
void ell_spmv(const ELLMatrix<hiscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y)
{
    auto dctx = x->get_device_context();

    // On halo stream: pack send buffer and copy to host if needed
    x->update_halos_pack_send_buffer(mat);

    // On compute stream: launch interior computations
    ell_interior_spmv(mat, x, y);

    // wait for comms to complete
    x->update_halos_send_receive(mat);
    x->update_halos_finalize(mat);

    ell_halo_spmv(mat, x, y);

    dctx->synchronize_halo_stream();
    dctx->synchronize_compute_stream();
}

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ell(const SparseMatrix_type& A, Vector_type& x, Vector_type& y)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    using scalar_type = typename SparseMatrix_type::scalar_type;
    auto elldata      = static_cast<const EllOptData<scalar_type, scalar_type>*>(A.optimizationData);
    ell_spmv(elldata->mat.get(), &x, &y);

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}

template int ComputeSPMV_ell< SparseMatrix<double>, Vector<double> >(
    const SparseMatrix<double>&, Vector<double>&, Vector<double>&);

template int ComputeSPMV_ell< SparseMatrix<float>, Vector<float> >(
    const SparseMatrix<float>&, Vector<float>&, Vector<float>&);
