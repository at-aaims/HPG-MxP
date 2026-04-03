/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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
 @file ComputeProlongation.cpp

 HPCG routine
 */

#include "prolongation.hpp"

#include "ell_matrix.hpp"

#include "kernel_helpers.hpp.inc"

template<unsigned int BLOCKSIZE, typename mat_scalar_type, typename vec_scalar_t>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_prolongation(const local_int_t size,
                                        const local_int_t* __restrict__ f2cOperator,
                                        const mat_scalar_type* __restrict__ coarse,
                                        vec_scalar_t* __restrict__ fine,
                                        const local_int_t* __restrict__ perm_fine,
                                        const local_int_t* __restrict__ perm_coarse)
{
    const local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx_coarse >= size) {
        return;
    }

    const local_int_t idx_fine = __ldcg(f2cOperator + idx_coarse);
    const local_int_t idx_perm = __ldcg(perm_coarse + idx_coarse);

    fine[perm_fine[idx_fine]] += coarse[idx_perm];
}

/*!
 * Adds the correction computed on the coarse grid to fine-grid solution vector.
 *
 * @param[in] Af  Fine grid sparse matrix object,
 *                containing pointers to current coarse grid correction and the f2c operator.
 * @param[inout] xf - Fine grid solution vector, update with coarse grid correction.
 *
 * Note that the fine grid residual is never explicitly constructed.
 * We only compute it for the fine grid points that will be injected into
 * corresponding coarse grid points.
 *
 * @return Returns zero on success and a non-zero value otherwise.
 */
template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int prolongation(const SparseMatrix<local_scalar_t, halo_scalar_t>& Af, Vector<vec_scalar_t>& xf)
{
    auto stream_interior = Af.dctx->get_compute_stream();
    dim3 blocks((Af.mgData->rc->local_length() - 1) / 128 + 1);
    dim3 threads(128);

    kernel_prolongation<128><<<blocks, threads, 0, stream_interior>>>(
        Af.mgData->rc->local_length(),
        Af.mgData->d_f2cOperator,
        Af.mgData->xc->d_values(),
        xf.d_values(),
        Af.perm,
        Af.Ac->perm);

    Af.dctx->synchronize_compute_stream();
    return 0;
}

template int prolongation(const SparseMatrix<double>& Af, Vector<double>& xf);
template int prolongation(const SparseMatrix<float>& Af, Vector<float>& xf);
template int prolongation(const SparseMatrix<double, float>& Af, Vector<float>& xf);
template int prolongation(const SparseMatrix<double, float>& Af, Vector<double>& xf);
