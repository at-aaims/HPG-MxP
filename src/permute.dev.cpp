/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * Modifications copyright (c) 2025 Oak Ridge National Laboratory.
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

/*!
 @file Permute.cpp

 HPCG routine
 */

#include "utils.hpp"
#include "Permute.hpp"

#include <hip/hip_runtime.h>

#define LAUNCH_PERM_COLS(blocksizex, blocksizey)                       \
    {                                                                  \
        dim3 blocks((A.localNumberOfRows - 1) / blocksizey + 1);       \
        dim3 threads(blocksizex, blocksizey);                          \
                                                                       \
        kernel_perm_cols<blocksizex, blocksizey><<<blocks, threads>>>( \
            A.localNumberOfRows,                                       \
            A.localNumberOfColumns,                                    \
            A.max_nnz_per_row,                                         \
            A.perm,                                                    \
            A.d_mtxIndL,                                               \
            A.d_matrixValues);                                         \
    }

template <unsigned int BLOCKSIZE, typename scalar>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_permute_ell_rows(const local_int_t m,
                                        const local_int_t p,
                                        const local_int_t* __restrict__ tmp_cols,
                                        const scalar* __restrict__ tmp_vals,
                                        const local_int_t* __restrict__ perm,
                                        local_int_t* __restrict__ ell_col_ind,
                                        scalar* __restrict__ ell_val)
{
    local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(row >= m)
    {
        return;
    }

    local_int_t idx = p * m + perm[row];
    local_int_t col = tmp_cols[row];

    ell_col_ind[idx] = col;
    ell_val[idx] = tmp_vals[row];
}

template <typename scalar>
__device__ void swap(local_int_t& key, scalar& val, int mask, int dir)
{
    local_int_t key1 = __shfl_xor(key, mask);
    scalar val1 = __shfl_xor(val, mask);

    if(key < key1 == dir)
    {
        key = key1;
        val = val1;
    }
}

__device__ int get_bit(int x, int i)
{
    return (x >> i) & 1;
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY, typename scalar>
__launch_bounds__(BLOCKSIZEX * BLOCKSIZEY)
__global__ void kernel_perm_cols(const local_int_t m,
                                 const local_int_t n,
                                 const local_int_t nonzerosPerRow,
                                 const local_int_t* __restrict__ perm,
                                 local_int_t* __restrict__ mtxIndL,
                                 scalar* __restrict__ matrixValues)
{
    const local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;
    const local_int_t idx = row * nonzerosPerRow + threadIdx.x;
    local_int_t key = n;
    scalar val = 0.0;

    if(threadIdx.x < nonzerosPerRow && row < m)
    {
        local_int_t col = mtxIndL[idx];
        val = matrixValues[idx];

        if(col >= 0 && col < m)
        {
            key = perm[col];
        }
        else if(col >= m && col < n)
        {
            key = col;
        }
    }

    swap(key, val, 1, get_bit(threadIdx.x, 1) ^ get_bit(threadIdx.x, 0));

    swap(key, val, 2, get_bit(threadIdx.x, 2) ^ get_bit(threadIdx.x, 1));
    swap(key, val, 1, get_bit(threadIdx.x, 2) ^ get_bit(threadIdx.x, 0));

    swap(key, val, 4, get_bit(threadIdx.x, 3) ^ get_bit(threadIdx.x, 2));
    swap(key, val, 2, get_bit(threadIdx.x, 3) ^ get_bit(threadIdx.x, 1));
    swap(key, val, 1, get_bit(threadIdx.x, 3) ^ get_bit(threadIdx.x, 0));

    swap(key, val, 8, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 3));
    swap(key, val, 4, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 2));
    swap(key, val, 2, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 1));
    swap(key, val, 1, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 0));

    swap(key, val, 16, get_bit(threadIdx.x, 4));
    swap(key, val,  8, get_bit(threadIdx.x, 3));
    swap(key, val,  4, get_bit(threadIdx.x, 2));
    swap(key, val,  2, get_bit(threadIdx.x, 1));
    swap(key, val,  1, get_bit(threadIdx.x, 0));

    if(threadIdx.x < nonzerosPerRow && row < m)
    {
        mtxIndL[idx] = (key == n) ? -1 : key;
        matrixValues[idx] = val;
    }
}

template <typename scalar>
void PermuteColumns(SparseMatrix<scalar>& A)
{
    // Determine blocksize in x direction
    unsigned int dim_x = A.max_nnz_per_row;

    // Compute next power of two
    dim_x |= dim_x >> 1;
    dim_x |= dim_x >> 2;
    dim_x |= dim_x >> 4;
    dim_x |= dim_x >> 8;
    dim_x |= dim_x >> 16;
    ++dim_x;

    // Determine blocksize
    unsigned int dim_y = 512 / dim_x;

    // Compute next power of two
    dim_y |= dim_y >> 1;
    dim_y |= dim_y >> 2;
    dim_y |= dim_y >> 4;
    dim_y |= dim_y >> 8;
    dim_y |= dim_y >> 16;
    ++dim_y;

    // Shift right until we obtain a valid blocksize
    while(dim_x * dim_y > 512)
    {
        dim_y >>= 1;
    }

    if     (dim_y == 32) LAUNCH_PERM_COLS(32, 32)
    else if(dim_y == 16) LAUNCH_PERM_COLS(32, 16)
    else if(dim_y ==  8) LAUNCH_PERM_COLS(32,  8)
    else                 LAUNCH_PERM_COLS(32,  4)
}

template <typename scalar>
void PermuteRows(SparseMatrix& A)
{
    const local_int_t m = A.localNumberOfRows;

    // Temporary structures for row permutation
    local_int_t* tmp_cols;
    scalar* tmp_vals;

    HIP_CHECK(deviceMalloc((void**)&tmp_cols, sizeof(local_int_t) * m));
    HIP_CHECK(deviceMalloc((void**)&tmp_vals, sizeof(scalar) * m));

    // Permute ELL rows
    for(local_int_t p = 0; p < A.ell_width; ++p)
    {
        const local_int_t offset = p * m;

        HIP_CHECK(hipMemcpy(tmp_cols, A.ell_col_ind + offset, sizeof(local_int_t) * m, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(tmp_vals, A.ell_val + offset, sizeof(scalar) * m, hipMemcpyDeviceToDevice));

        kernel_permute_ell_rows<1024><<<(m - 1) / 1024 + 1, 1024>>>(
            m,
            p,
            tmp_cols,
            tmp_vals,
            A.perm,
            A.ell_col_ind,
            A.ell_val);
    }

    HIP_CHECK(deviceFree(tmp_cols));
    HIP_CHECK(deviceFree(tmp_vals));
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_permute(local_int_t size,
                               const local_int_t* __restrict__ perm,
                               const scalar* __restrict__ in,
                               scalar* __restrict__ out)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= size)
    {
        return;
    }

    out[perm[gid]] = in[gid];
}

template <typename scalar>
void PermuteVector(local_int_t size, Vector& v, const local_int_t* perm)
{
    scalar* buffer;
    HIP_CHECK(deviceMalloc((void**)&buffer, sizeof(scalar) * v.localLength));

    kernel_permute<1024><<<(size - 1) / 1024 + 1, 1024>>>(size, perm, v.d_values, buffer);

    HIP_CHECK(deviceFree(v.d_values));
    v.d_values = buffer;
}
