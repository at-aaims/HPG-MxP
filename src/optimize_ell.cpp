/*!
 * @file optimize_ell.cpp
 * @copyright Modifications (c) 2019 Advanced Micro Devices, Inc.,
 *            further modifications (c) 2025 Oak Ridge National Laboratory.
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

#if !defined HPGMP_REFERENCE

//#include "optimize_ell.hpp"

#include <memory>
#include <iostream>

#include "SparseMatrix.hpp"
#include "ell_matrix.hpp"
#include "GMRESData.hpp"
//#include "Permute.hpp"
//#include "multiColoring.hpp"

/*!
  Optimizes the data structures used for GMRES to increase the
  performance of the benchmark version of the preconditioned GMRES algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with necessary GMRES vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
template <typename mat_scalar, typename solver_scalar, typename vec_scalar>
int OptimizeProblemELL(SparseMatrix<mat_scalar>& A, GMRESData<solver_scalar>& data,
                       Vector<vec_scalar>& b, Vector<vec_scalar>& x, Vector<vec_scalar>& xexact)
{
    auto dctx = A.dctx;

    // Perform matrix coloring
    //JPLColoring(A);

    // Permute matrix columns
    //PermuteColumns(A);

    // Convert matrix to ELL format
    //ConvertToELL(A);
    auto aoptdata = new EllOptData<mat_scalar>;// static_cast<EllOptData<mat_scalar>*>(A.optimizationData);
    aoptdata->mat = std::make_shared<ELLMatrix<mat_scalar>>(A);
    A.optimizationData = aoptdata;
#ifdef HPGMP_VERBOSE
    MPI_Barrier(A.comm);
    if(A.geom->rank == 0) {
        std::cout << "Built fine-grid ELL." << std::endl;
    }
#endif

    // Defrag permutation vector
    //HIP_CHECK(deviceDefrag((void**)&A.perm, sizeof(local_int_t) * A.localNumberOfRows));

    // Permute matrix rows
    //PermuteRows(A);

    // Extract diagonal indices and inverse values
    //ExtractDiagonal(A);

    // Defrag
#ifdef HPGMP_MEMORY_MANAGEMENT
    //HIP_CHECK(deviceDefrag((void**)&A.diag_idx, sizeof(local_int_t) * A.localNumberOfRows));
    //HIP_CHECK(deviceDefrag((void**)&A.inv_diag, sizeof(double) * A.localNumberOfRows));
#ifndef HPGMP_NO_MPI
    //HIP_CHECK(deviceDefrag((void**)&A.d_send_buffer, sizeof(double) * A.totalToBeSent));
    //HIP_CHECK(deviceDefrag((void**)&A.d_elementsToSend, sizeof(local_int_t) * A.totalToBeSent));
#endif
#endif

    // Permute vectors
    //PermuteVector(A.localNumberOfRows, b, A.perm);
    //PermuteVector(A.localNumberOfRows, xexact, A.perm);

    // Initialize GMRES structures
    //data.initialize(A, dctx);

    // Process all coarse level matrices
    SparseMatrix<mat_scalar>* M = A.Ac;

    int igrid = 1;
    while(M != NULL)
    {
        // Perform matrix coloring
        //JPLColoring(*M);

        // Permute matrix columns
        //PermuteColumns(*M);

        // Convert matrix to ELL format
        //ConvertToELL(*M);
        auto moptdata = new EllOptData<mat_scalar>;
        moptdata->mat = std::make_shared<ELLMatrix<mat_scalar>>(*M);
        M->optimizationData = moptdata;
#ifdef HPGMP_VERBOSE
        MPI_Barrier(A.comm);
        if(A.geom->rank == 0) {
            std::cout << "Built ELL on grid " << igrid << "." << std::endl;
        }
#endif

        // Defrag matrix arrays and permutation vector
        //HIP_CHECK(deviceDefrag((void**)&M->ell_col_ind,
        //                       sizeof(local_int_t) * M->ell_width * M->localNumberOfRows));
        //HIP_CHECK(deviceDefrag((void**)&M->ell_val, sizeof(double) * M->ell_width * M->localNumberOfRows));
        //HIP_CHECK(deviceDefrag((void**)&M->perm, sizeof(local_int_t) * M->localNumberOfRows));

        // Permute matrix rows
        //PermuteRows(*M);

        // Extract diagonal indices and inverse values
        //ExtractDiagonal(*M);

        // Defrag
#ifdef HPGMP_MEMORY_MANAGEMENT
        //HIP_CHECK(deviceDefrag((void**)&M->diag_idx, sizeof(local_int_t) * M->localNumberOfRows));
        //HIP_CHECK(deviceDefrag((void**)&M->inv_diag, sizeof(double) * M->localNumberOfRows));
#ifndef HPGMP_NO_MPI
        //HIP_CHECK(deviceDefrag((void**)&M->d_send_buffer, sizeof(double) * M->totalToBeSent));
        //HIP_CHECK(deviceDefrag((void**)&M->d_elementsToSend, sizeof(local_int_t) * M->totalToBeSent));
#endif
#endif

        // Go to next level in hierarchy
        M = M->Ac;
        igrid++;
    }

    // Defrag hierarchy structures
#ifdef HPGMP_MEMORY_MANAGEMENT
    //M = &A;
    //MGData* mg = M->mgData;

    //while(mg != NULL)
    //{
    //    M = M->Ac;

    //    HIP_CHECK(deviceDefrag((void**)&mg->d_f2cOperator, sizeof(local_int_t) * M->localNumberOfRows));
    //    HIP_CHECK(deviceDefrag((void**)&mg->rc->d_values, sizeof(double) * mg->rc->localLength));
    //    HIP_CHECK(deviceDefrag((void**)&mg->xc->d_values, sizeof(double) * mg->xc->localLength));

    //    mg = M->mgData;
    //}
#endif

    if(A.geom->rank == 0) {
        std::cout << "Finished building all ELL matrices." << std::endl;
    }
    return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
//double OptimizeProblemMemoryUse(const SparseMatrix & A)
//{
//    return 0.0;
//}

template
int OptimizeProblemELL(SparseMatrix<double>& A, GMRESData<double>& data,
                    Vector<double>& b, Vector<double>& x, Vector<double>& xexact);

template
int OptimizeProblemELL(SparseMatrix<float>& A, GMRESData<float>& data,
                    Vector<float>& b, Vector<float>& x, Vector<float>& xexact);

template
int OptimizeProblemELL(SparseMatrix<float>& A, GMRESData<double>& data,
                    Vector<double>& b, Vector<double>& x, Vector<double>& xexact);

#endif // HPGMP_REFERENCE
