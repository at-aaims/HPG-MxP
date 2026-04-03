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

#include <memory>
#include <iostream>

#include "SparseMatrix.hpp"
#include "ell_matrix.hpp"
#include "GMRESData.hpp"
#include "permute.hpp"
#include "multicoloring.hpp"

#ifdef HPGMP_WITH_GINKGO
#include "GinkgoOptData.hpp"
#endif

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
template<typename local_scalar_t, typename halo_scalar_t, class GMRESData_type, typename vec_scalar_type>
int OptimizeProblemELL(SparseMatrix<local_scalar_t, halo_scalar_t>& A, GMRESData_type& data,
                       Vector<vec_scalar_type>& b, Vector<vec_scalar_type>& x, Vector<vec_scalar_type>& xexact)
{
    b.update_device_data();
    xexact.update_device_data();

    auto dctx             = A.dctx;
    const int local_nrows = A.localNumberOfRows;

    SparseMatrix<local_scalar_t, halo_scalar_t>* M = &A;
    int igrid                                      = 0;
    while (M != NULL)
    {

        // Allocate device copies of data needed for ELL and RBGS
        const auto local_nrows = M->localNumberOfRows;
        M->d_mtxIndL           = reinterpret_cast<local_int_t*>(
            dctx->device_alloc(M->max_nnz_per_row * local_nrows * sizeof(local_int_t)));
        dctx->copy_host_to_device_sync(M->d_mtxIndL, M->mtxIndL[0],
                                       M->max_nnz_per_row * local_nrows * sizeof(local_int_t));
        M->d_matrixValues = reinterpret_cast<local_scalar_t*>(
            dctx->device_alloc(M->max_nnz_per_row * local_nrows * sizeof(local_scalar_t)));
        dctx->copy_host_to_device_sync(M->d_matrixValues, M->matrixValues[0],
                                       M->max_nnz_per_row * local_nrows * sizeof(local_scalar_t));

        // Perform matrix coloring
        multicolor_JPL(*M);

        // Permute matrix columns
        permute_columns(*M);

        // Convert matrix to ELL format
#ifdef HPGMP_VERBOSE
        if (A.geom->rank == 0) {
            std::cout << "Setting up ELL on grid " << igrid << "." << std::endl;
        }
#endif
#ifdef HPGMP_WITH_GINKGO
        auto moptdata       = new GinkgoOptData<local_scalar_t, halo_scalar_t>;
        auto mat            = std::make_shared<GinkgoMatrix<local_scalar_t, halo_scalar_t>>(*M);
        moptdata->mat       = mat;
        moptdata->solver    = std::make_shared<GinkgoSolver<local_scalar_t, halo_scalar_t>>(mat.get());
        M->optimizationData = moptdata;
#else
        auto moptdata       = new EllOptData<local_scalar_t, halo_scalar_t>;
        moptdata->mat       = std::make_shared<ELLMatrix<local_scalar_t, halo_scalar_t>>(*M); // Performs row permutation
        M->optimizationData = moptdata;
#endif
#ifdef HPGMP_VERBOSE
        MPI_Barrier(A.comm);
        if (A.geom->rank == 0) {
            std::cout << "Built ELL on grid " << igrid << "." << std::endl;
        }
#endif

        // Go to next level in hierarchy
        M = M->Ac;
        igrid++;
    }

    // Permute vectors
    b.permute(A.perm);
    xexact.permute(A.perm);

    // Update host mirrors with permutted values
    b.update_host_mirror();
    xexact.update_host_mirror();

    return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
//double OptimizeProblemMemoryUse(const SparseMatrix & A)
//{
//    return 0.0;
//}

template int OptimizeProblemELL(SparseMatrix<double>& A, GMRESData<double, double, double>& data,
                                Vector<double>& b, Vector<double>& x, Vector<double>& xexact);

template int OptimizeProblemELL(SparseMatrix<float>& A, GMRESData<float, float, float>& data,
                                Vector<float>& b, Vector<float>& x, Vector<float>& xexact);

template int OptimizeProblemELL(SparseMatrix<float>& A, GMRESData<double, double, double>& data,
                                Vector<double>& b, Vector<double>& x, Vector<double>& xexact);

template int OptimizeProblemELL(SparseMatrix<double, float>& A, GMRESData<double, float, float>& data,
                                Vector<float>& b, Vector<float>& x, Vector<float>& xexact);

template int OptimizeProblemELL(SparseMatrix<double, float>& A, GMRESData<double, double, double>& data,
                                Vector<double>& b, Vector<double>& x, Vector<double>& xexact);

#if 0
template <typename scalar_t>
void seemingly_necessary_stuff_from_reference(SparseMatrix<scalar_t>* M)
{
    auto dctx = M->dctx;
    SparseMatrix<scalar_t> *curLevelMatrix = M;
    do {
      // form CSR on host
      const local_int_t nrow = curLevelMatrix->localNumberOfRows;
      const local_int_t ncol = curLevelMatrix->localNumberOfColumns;
      global_int_t nnzL = 0;
      global_int_t nnz = curLevelMatrix->localNumberOfNonzeros;
      int *h_row_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_col_idx = (int*)malloc( nnz    * sizeof(int));
      scalar_t  *h_nzvals  = (scalar_t *)malloc( nnz    * sizeof(scalar_t));

      nnz = 0;
      h_row_ptr[0] = 0;
      for (local_int_t i=0; i<nrow; i++)  {
        const scalar_t * const cur_vals = curLevelMatrix->matrixValues[i];
        const local_int_t * const cur_inds = curLevelMatrix->mtxIndL[i];

        const local_int_t cur_nnz = curLevelMatrix->nonzerosInRow[i];
        for (local_int_t j=0; j<cur_nnz; j++) {
          h_nzvals[nnz+j] = cur_vals[j];
          h_col_idx[nnz+j] = cur_inds[j];
          if (cur_inds[j] <= i) {
            nnzL++;
          }
        }
        nnz += cur_nnz;
        h_row_ptr[i+1] = nnz;
      }
      const global_int_t totalToBeSent = curLevelMatrix->totalToBeSent;

      // copy CSR(A) to device
      curLevelMatrix->d_row_ptr = static_cast<int*>(dctx->device_alloc((nrow+1)*sizeof(int)));
      curLevelMatrix->d_col_idx = static_cast<int*>(dctx->device_alloc(nnz*sizeof(int)));
      curLevelMatrix->d_nzvals = static_cast<scalar_t>(dctx->device_alloc(nnz*sizeof(scalar_t)));
#ifndef HPGMP_NO_MPI
      curLevelMatrix->d_sendBuffer = static_cast<scalar_t>(dctx->device_alloc(nnz*sizeof(scalar_t)));
      curLevelMatrix->d_elementsToSend = static_cast<local_int_t*>(
                                    dctx->device_alloc(totalToBeSent*sizeof(local_int_t)));
#endif
      dctx->copy_host_to_device_sync(curLevelMatrix->d_row_ptr, h_row_ptr, (nrow+1)*sizeof(int));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_col_idx, h_col_idx, nnz*sizeof(int));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_nzvals, h_nzvals, nnz*sizeof(scalar_t));
#ifndef HPGMP_NO_MPI
      dctx->copy_host_to_device_sync(curLevelMatrix->d_elementsToSend,
              curLevelMatrix->elementsToSend, totalToBeSent*sizeof(local_int_t));
#endif
      // free matrix on host
      free(h_row_ptr);
      free(h_col_idx);
      free(h_nzvals);

      // Extract lower/upper-triangular matrix
      global_int_t nnzU = nnz-nnzL;
      int *h_Lrow_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_Lcol_idx = (int*)malloc( nnzL   * sizeof(int));
      scalar_t  *h_Lnzvals  = (scalar_t *)malloc( nnzL   * sizeof(scalar_t));
      int *h_Urow_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_Ucol_idx = (int*)malloc( nnzU   * sizeof(int));
      scalar_t  *h_Unzvals  = (scalar_t *)malloc( nnzU   * sizeof(scalar_t));
      nnzL = 0;
      nnzU = 0;
      h_Lrow_ptr[0] = 0;
      h_Urow_ptr[0] = 0;
      for (local_int_t i=0; i<nrow; i++)  {
        const scalar_t * const cur_vals = curLevelMatrix->matrixValues[i];
        const local_int_t * const cur_inds = curLevelMatrix->mtxIndL[i];

        const int cur_nnz = curLevelMatrix->nonzerosInRow[i];
        for (int j=0; j<cur_nnz; j++) {
          if (cur_inds[j] <= i) {
            h_Lnzvals[nnzL] = cur_vals[j];
            h_Lcol_idx[nnzL] = cur_inds[j];
            nnzL ++;
          } else {
            h_Unzvals[nnzU] = cur_vals[j];
            h_Ucol_idx[nnzU] = cur_inds[j];
            nnzU ++;
          }
        }
        h_Lrow_ptr[i+1] = nnzL;
        h_Urow_ptr[i+1] = nnzU;
      }
      curLevelMatrix->nnzL = nnzL;
      curLevelMatrix->nnzU = nnzU;

      // copy CSR(L) to device
      curLevelMatrix->d_Lrow_ptr = static_cast<int*>(dctx->device_alloc((nrow+1)*sizeof(int)));
      curLevelMatrix->d_Lcol_idx = static_cast<int*>(dctx->device_alloc(nnzL*sizeof(int)));
      curLevelMatrix->d_Lnzvals = static_cast<scalar_t>(dctx->device_alloc(nnzL*sizeof(scalar_t)));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_Lrow_ptr, h_Lrow_ptr, (nrow+1)*sizeof(int));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_Lcol_idx, h_Lcol_idx, nnzL*sizeof(int));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_Lnzvals, h_Lnzvals, nnzL*sizeof(scalar_t));

      // copy CSR(U) to device
      curLevelMatrix->d_Urow_ptr = static_cast<int*>(dctx->device_alloc((nrow+1)*sizeof(int)));
      curLevelMatrix->d_Ucol_idx = static_cast<int*>(dctx->device_alloc(nnzU*sizeof(int)));
      curLevelMatrix->d_Unzvals = static_cast<scalar_t>(dctx->device_alloc(nnzU*sizeof(scalar_t)));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_Urow_ptr, h_Urow_ptr, (nrow+1)*sizeof(int));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_Ucol_idx, h_Ucol_idx, nnzU*sizeof(int));
      dctx->copy_host_to_device_sync(curLevelMatrix->d_Unzvals,  h_Unzvals, nnzU*sizeof(scalar_t));

      // free matrix on host
      free(h_Lrow_ptr);
      free(h_Lcol_idx);
      free(h_Lnzvals);
      free(h_Urow_ptr);
      free(h_Ucol_idx);
      free(h_Unzvals);

      // Clean up unnecessary things from reference optimize problem
      auto M = curLevelMatrix;
      dctx->device_free(M->d_row_ptr); M->d_row_ptr = nullptr;
      dctx->device_free(M->d_col_idx); M->d_col_idx = nullptr;
      dctx->device_free(M->d_nzvals); M->d_nzvals = nullptr;
      dctx->device_free(M->d_elementsToSend); M->d_elementsToSend = nullptr;
      dctx->device_free(M->d_sendBuffer); M->d_sendBuffer = nullptr;
      dctx->device_free(M->d_Lrow_ptr); M->d_Lrow_ptr = nullptr;
      dctx->device_free(M->d_Lcol_idx); M->d_Lcol_idx = nullptr;
      dctx->device_free(M->d_Lnzvals); M->d_Lnzvals = nullptr;
      dctx->device_free(M->d_Urow_ptr); M->d_Urow_ptr = nullptr;
      dctx->device_free(M->d_Ucol_idx); M->d_Ucol_idx = nullptr;
      dctx->device_free(M->d_Unzvals); M->d_Unzvals = nullptr;

      curLevelMatrix = curLevelMatrix->Ac;
    } while (curLevelMatrix != nullptr);
}
#endif

// Helper function (see OptimizeProblem.hpp for details)
template<class SparseMatrix_type>
double OptimizeProblemMemoryUse(const SparseMatrix_type& A)
{
    return 0.0;
}

template double OptimizeProblemMemoryUse(const SparseMatrix<double>&);
template double OptimizeProblemMemoryUse(const SparseMatrix<float>&);
template double OptimizeProblemMemoryUse(const SparseMatrix<double, float>&);
