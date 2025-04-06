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

/** This routine repeats some things from the reference optimization,
 * but deletes any memory allocated in the process.
 */
template <typename mat_scalar>
void seemingly_necessary_stuff_from_reference(SparseMatrix<mat_scalar>* M);

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
    b.update_device_data();
    xexact.update_device_data();

    auto dctx = A.dctx;
    const int local_nrows = A.localNumberOfRows;

    seemingly_necessary_stuff_from_reference(&A);

    SparseMatrix<mat_scalar>* M = &A;
    int igrid = 0;
    while(M != NULL)
    {

        // Allocate device copies of data needed for ELL and RBGS
        const auto local_nrows = M->localNumberOfRows;
        M->d_mtxIndL = reinterpret_cast<local_int_t*>(
                dctx->device_alloc(M->max_nnz_per_row*local_nrows*sizeof(local_int_t)));
        dctx->copy_host_to_device_sync(M->d_mtxIndL, M->mtxIndL[0],
                M->max_nnz_per_row*local_nrows*sizeof(local_int_t));
        M->d_matrixValues = reinterpret_cast<mat_scalar*>(
                dctx->device_alloc(M->max_nnz_per_row*local_nrows*sizeof(mat_scalar)));
        dctx->copy_host_to_device_sync(M->d_matrixValues, M->matrixValues[0],
                M->max_nnz_per_row*local_nrows*sizeof(mat_scalar));

        // Perform matrix coloring
        multicolor_JPL(*M);

        // Permute matrix columns
        permute_columns(*M);

        // Convert matrix to ELL format
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
        moptdata->mat->permute_rows(M->perm);

        // Extract diagonal indices and inverse values
        moptdata->mat->extract_diagonal();

        // Defrag
#ifdef HPGMP_MEMORY_MANAGEMENT
        //HIP_CHECK(deviceDefrag((void**)&M->diag_idx, sizeof(local_int_t) * M->localNumberOfRows));
        //HIP_CHECK(deviceDefrag((void**)&M->inv_diag, sizeof(double) * M->localNumberOfRows));
#ifndef HPGMP_NO_MPI
        //HIP_CHECK(deviceDefrag((void**)&M->d_send_buffer, sizeof(double) * M->totalToBeSent));
        //HIP_CHECK(deviceDefrag((void**)&M->d_elementsToSend, sizeof(local_int_t) * M->totalToBeSent));
#endif
#endif

        dctx->device_free(M->d_mtxIndL);
        dctx->device_free(M->d_matrixValues);

        // Go to next level in hierarchy
        M = M->Ac;
        igrid++;
    }

    // Permute vectors
    //PermuteVector(A.localNumberOfRows, b, A.perm);
    //PermuteVector(A.localNumberOfRows, xexact, A.perm);
    b.permute(A.perm);
    xexact.permute(A.perm);

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
        std::cout << "Finished building, reordering and preparing all ELL matrices." << std::endl;
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

template <typename SC>
void seemingly_necessary_stuff_from_reference(SparseMatrix<SC>* M)
{
    auto dctx = M->dctx;
    SparseMatrix<SC> *curLevelMatrix = M;
    do {
      // form CSR on host
      const local_int_t nrow = curLevelMatrix->localNumberOfRows;
      const local_int_t ncol = curLevelMatrix->localNumberOfColumns;
      global_int_t nnzL = 0;
      global_int_t nnz = curLevelMatrix->localNumberOfNonzeros;
      int *h_row_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_col_ind = (int*)malloc( nnz    * sizeof(int));
      SC  *h_nzvals  = (SC *)malloc( nnz    * sizeof(SC));

      nnz = 0;
      h_row_ptr[0] = 0;
      for (local_int_t i=0; i<nrow; i++)  {
        const SC * const cur_vals = curLevelMatrix->matrixValues[i];
        const local_int_t * const cur_inds = curLevelMatrix->mtxIndL[i];

        const int cur_nnz = curLevelMatrix->nonzerosInRow[i];
        for (int j=0; j<cur_nnz; j++) {
          h_nzvals[nnz+j] = cur_vals[j];
          h_col_ind[nnz+j] = cur_inds[j];
          if (cur_inds[j] <= i) {
            nnzL++;
          }
        }
        nnz += cur_nnz;
        h_row_ptr[i+1] = nnz;
      }
      const global_int_t totalToBeSent = curLevelMatrix->totalToBeSent;

      // copy CSR(A) to device
#if defined(HPGMP_WITH_CUDA)
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_row_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_row_ptr(nrow=%d)\n",nrow );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_col_idx), nnz*sizeof(int))) {
        printf( " Failed to allocate A.d_col_idx(nnz=%lld)\n",nnz );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_nzvals),  nnz*sizeof(SC))) {
        printf( " Failed to allocate A.d_nzvals(nnz=%lld)\n",nnz );
      }
  #ifndef HPGMP_NO_MPI
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_sendBuffer), totalToBeSent*sizeof(SC))) {
        printf( " Failed to allocate A.d_sendBuffer(totalToBeSent=%lld)\n",totalToBeSent );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_elementsToSend), totalToBeSent*sizeof(local_int_t))) {
        printf( " Failed to allocate A.d_elementsToSend(totalToBeSent=%lld)\n",totalToBeSent );
      }
  #endif


      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_row_ptr, h_row_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_row_ptr\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_col_idx, h_col_ind, nnz*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_col_idx\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_nzvals,  h_nzvals,  nnz*sizeof(SC),  cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_nzvals\n" );
      }
  #ifndef HPGMP_NO_MPI
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_elementsToSend, curLevelMatrix->elementsToSend, totalToBeSent*sizeof(local_int_t), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_elementsToSend\n" );
      }
  #endif
      // free matrix on host
      free(h_row_ptr);
      free(h_col_ind);
      free(h_nzvals);
#elif defined(HPGMP_WITH_HIP)
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_row_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_row_ptr(nrow=%d)\n",nrow );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_col_idx), nnz*sizeof(int))) {
        printf( " Failed to allocate A.d_col_idx(nnz=%lld)\n",nnz );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_nzvals),  nnz*sizeof(SC))) {
        printf( " Failed to allocate A.d_nzvals(nnz=%lld)\n",nnz );
      }
  #ifndef HPGMP_NO_MPI
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_sendBuffer),  nnz*sizeof(SC))) {
        printf( " Failed to allocate A.d_sendBuffer(totalToBeSent=%lld)\n",totalToBeSent );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_elementsToSend), totalToBeSent*sizeof(local_int_t))) {
        printf( " Failed to allocate A.d_elementsToSend(totalToBeSent=%lld)\n",totalToBeSent );
      }
  #endif

      if (hipSuccess != hipMemcpy(curLevelMatrix->d_row_ptr, h_row_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_row_ptr\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_col_idx, h_col_ind, nnz*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_col_idx\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_nzvals,  h_nzvals,  nnz*sizeof(SC), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_nzvals\n" );
      }
  #ifndef HPGMP_NO_MPI
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_elementsToSend, curLevelMatrix->elementsToSend, totalToBeSent*sizeof(local_int_t), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_elementsToSend\n" );
      }
  #endif
      // free matrix on host
      free(h_row_ptr);
      free(h_col_ind);
      free(h_nzvals);
#endif

      // Extract lower/upper-triangular matrix
      global_int_t nnzU = nnz-nnzL;
      int *h_Lrow_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_Lcol_ind = (int*)malloc( nnzL   * sizeof(int));
      SC  *h_Lnzvals  = (SC *)malloc( nnzL   * sizeof(SC));
      int *h_Urow_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_Ucol_ind = (int*)malloc( nnzU   * sizeof(int));
      SC  *h_Unzvals  = (SC *)malloc( nnzU   * sizeof(SC));
      nnzL = 0;
      nnzU = 0;
      h_Lrow_ptr[0] = 0;
      h_Urow_ptr[0] = 0;
      for (local_int_t i=0; i<nrow; i++)  {
        const SC * const cur_vals = curLevelMatrix->matrixValues[i];
        const local_int_t * const cur_inds = curLevelMatrix->mtxIndL[i];

        const int cur_nnz = curLevelMatrix->nonzerosInRow[i];
        for (int j=0; j<cur_nnz; j++) {
          if (cur_inds[j] <= i) {
            h_Lnzvals[nnzL] = cur_vals[j];
            h_Lcol_ind[nnzL] = cur_inds[j];
            nnzL ++;
          } else {
            h_Unzvals[nnzU] = cur_vals[j];
            h_Ucol_ind[nnzU] = cur_inds[j];
            nnzU ++;
          }
        }
        h_Lrow_ptr[i+1] = nnzL;
        h_Urow_ptr[i+1] = nnzU;
      }
      curLevelMatrix->nnzL = nnzL;
      curLevelMatrix->nnzU = nnzU;

      // copy CSR(L) to device
  #if defined(HPGMP_WITH_CUDA)
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Lrow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Lcol_idx), nnzL*sizeof(int))) {
        printf( " Failed to allocate A.d_Lcol_idx\n" );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Lnzvals),  nnzL*sizeof(SC))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }

      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Lrow_ptr, h_Lrow_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Lcol_idx, h_Lcol_ind, nnzL*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lcol_idx\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Lnzvals,  h_Lnzvals,  nnzL*sizeof(SC),  cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
  #elif defined(HPGMP_WITH_HIP)
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Lrow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Lcol_idx), nnzL*sizeof(int))) {
        printf( " Failed to allocate A.d_Lcol_idx\n" );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Lnzvals),  nnzL*sizeof(SC))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }

      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Lrow_ptr, h_Lrow_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Lcol_idx, h_Lcol_ind, nnzL*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lcol_idx\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Lnzvals,  h_Lnzvals,  nnzL*sizeof(SC),  hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
  #endif

      // copy CSR(U) to device
  #if defined(HPGMP_WITH_CUDA)
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Urow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Urow_ptr(nrow=%d)\n",nrow );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Ucol_idx), nnzU*sizeof(int))) {
        printf( " Failed to allocate A.d_Ucol_idx(nnzU=%d)\n",nnzU );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Unzvals),  nnzU*sizeof(SC))) {
        printf( " Failed to allocate A.d_Urow_ptr(nnzU=%d)\n",nnzU );
      }

      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Urow_ptr, h_Urow_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Ucol_idx, h_Ucol_ind, nnzU*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Ucol_idx\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Unzvals,  h_Unzvals,  nnzU*sizeof(SC),  cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
  #elif defined(HPGMP_WITH_HIP)
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Urow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Urow_ptr(nrow=%d)\n",nrow );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Ucol_idx), nnzU*sizeof(int))) {
        printf( " Failed to allocate A.d_Ucol_idx(nnzU=%lld)\n",nnzU );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Unzvals),  nnzU*sizeof(SC))) {
        printf( " Failed to allocate A.d_Urow_ptr(nnzU=%lld)\n",nnzU );
      }

      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Urow_ptr, h_Urow_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Ucol_idx, h_Ucol_ind, nnzU*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Ucol_idx\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Unzvals,  h_Unzvals,  nnzU*sizeof(SC),  hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
  #endif

      // free matrix on host
      free(h_Lrow_ptr);
      free(h_Lcol_ind);
      free(h_Lnzvals);
      free(h_Urow_ptr);
      free(h_Ucol_ind);
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

