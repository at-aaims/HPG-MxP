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

#include "permute.hpp"

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

template<typename scalar_type>
__device__ void swap(local_int_t& key, scalar_type& val, int mask, int dir)
{
#ifdef HPGMP_WITH_HIP
    const local_int_t key1 = __shfl_xor(key, mask);
    const scalar_type val1 = __shfl_xor(val, mask);
#elif HPGMP_WITH_CUDA
    const local_int_t key1 = __shfl_xor_sync(0xffffffff, key, mask);
    const scalar_type val1 = __shfl_xor_sync(0xffffffff, val, mask);
#endif

    if (key < key1 == dir)
    {
        key = key1;
        val = val1;
    }
}

__device__ int get_bit(int x, int i)
{
    return (x >> i) & 1;
}

template<unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY, typename scalar_type>
__launch_bounds__(BLOCKSIZEX* BLOCKSIZEY)
    __global__ void kernel_perm_cols(const local_int_t m,
                                     const local_int_t n,
                                     const local_int_t nonzerosPerRow,
                                     const local_int_t* __restrict__ perm,
                                     local_int_t* __restrict__ mtxIndL,
                                     scalar_type* __restrict__ matrixValues)
{
    const local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;
    const local_int_t idx = row * nonzerosPerRow + threadIdx.x;
    local_int_t key       = n;
    scalar_type val       = 0.0;

    if (threadIdx.x < nonzerosPerRow && row < m)
    {
        local_int_t col = mtxIndL[idx];
        val             = matrixValues[idx];

        if (col >= 0 && col < m)
        {
            key = perm[col];
        } else if (col >= m && col < n)
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
    swap(key, val, 8, get_bit(threadIdx.x, 3));
    swap(key, val, 4, get_bit(threadIdx.x, 2));
    swap(key, val, 2, get_bit(threadIdx.x, 1));
    swap(key, val, 1, get_bit(threadIdx.x, 0));

    if (threadIdx.x < nonzerosPerRow && row < m)
    {
        mtxIndL[idx]      = (key == n) ? -1 : key;
        matrixValues[idx] = val;
    }
}

template<typename scalar_type>
void permute_columns(SparseMatrix<scalar_type>& A)
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
    while (dim_x * dim_y > 512)
    {
        dim_y >>= 1;
    }

    if (dim_y == 32) LAUNCH_PERM_COLS(32, 32)
    else if (dim_y == 16) LAUNCH_PERM_COLS(32, 16)
    else if (dim_y == 8) LAUNCH_PERM_COLS(32, 8)
    else LAUNCH_PERM_COLS(32, 4)
}

template void permute_columns(SparseMatrix<float>& A);
template void permute_columns(SparseMatrix<double>& A);
