
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeGS_Forward_ref.cpp

 HPGMP routine
 */
#if (defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP))

#include <cassert>
#include <iostream>

#ifndef HPGMP_NO_MPI
 #include "ExchangeHalo.hpp"
#endif
#include "ComputeGS_Forward_ref.hpp"

#include "DataTypes.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeWAXPBY.hpp"
#ifdef HPGMP_DEBUG
 #include <mpi.h>
 #include "Utils_MPI.hpp"
 #include "hpgmp.hpp"
#endif
#include "mytimer.hpp"

/*!
  Computes one forward step of Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeGS_Forward
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeGS_Forward_ref(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x) {

  assert(x.local_length()==A.localNumberOfColumns); // Make sure x contain space for halo values

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  const local_int_t nrow = A.localNumberOfRows;
  const local_int_t ncol = A.localNumberOfColumns;

  scalar_type * const xv = x.values();

  // workspace
  scalar_type * const d_bv = A.workx.d_values();
  scalar_type * const d_xv = x.d_values();

#ifndef HPGMP_NO_MPI

  double time1 = x.time1;
  double time2 = x.time2;

  // Exchange Halo
  #ifdef HPGMP_WITH_CUDA
  // Copy local part of X to HOST CPU
  if (cudaSuccess != cudaMemcpy(xv, x.d_values(), nrow*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #elif defined(HPGMP_WITH_HIP)
  if (hipSuccess != hipMemcpy(xv, x.d_values(), nrow*sizeof(scalar_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #endif

  ExchangeHalo(A, x);

  // copy non-local part of X to device (after Halo exchange)
  #if defined(HPGMP_WITH_CUDA)
  if (cudaSuccess != cudaMemcpy(d_xv + nrow, &xv[nrow], (ncol-nrow)*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #elif defined(HPGMP_WITH_HIP)
  if (hipSuccess != hipMemcpy(d_xv + nrow, &xv[nrow], (ncol-nrow)*sizeof(scalar_type), hipMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #endif
  // restore timers (Exchange record comm in those)
  x.time1 = time1; x.time2 = time2;

  #ifdef HPGMP_DEBUG
  if (A.geom->rank==0) {
    HPGMP_fout << A.geom->rank << " : ComputeGS(" << nrow << " x " << ncol << ") start" << std::endl;
  }
  #endif
#endif

#if defined(HPGMP_DEBUG)
  const scalar_type * const rv = r.values();
  scalar_type ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues

  for (local_int_t i=0; i < nrow; i++) {
    const scalar_type * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const scalar_type currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    scalar_type sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }
#endif

  const scalar_type  one ( 1.0);
  const scalar_type mone (-1.0);
  double t0 = 0.0;

  // b = r - Ux
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(d_bv, r.d_values(), nrow*sizeof(scalar_type), cudaMemcpyDeviceToDevice)) {
      printf( " Failed to memcpy d_r\n" );
    }
    cusparseStatus_t status;
    #if CUDA_VERSION >= 11000
    cudaDataType computeType;
    if (std::is_same<scalar_type, double>::value) {
      computeType = CUDA_R_64F;
    } else if (std::is_same<scalar_type, float>::value) {
      computeType = CUDA_R_32F;
    }
    // create matrix
    cusparseSpMatDescr_t A_cusparse;
    cusparseCreateCsr(&A_cusparse, nrow, ncol, A.nnzU,
                      A.d_Urow_ptr, A.d_Ucol_idx, A.d_Unzvals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, computeType);
    // create vectors
    cusparseDnVecDescr_t vecX, vecB;
    cusparseCreateDnVec(&vecX, ncol, d_xv, computeType);
    cusparseCreateDnVec(&vecB, nrow, d_bv, computeType);
    // SpMV
    TICK();
    status = cusparseSpMV(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &mone, A_cusparse,
                                 vecX,
                          &one,  vecB,
                          computeType, CUSPARSE_MV_ALG_DEFAULT, A.buffer_U);
    TOCK(x.time1);
    if (CUSPARSE_STATUS_SUCCESS != status) {
      printf( " Failed cusparseSpMV for GS\n" );
    }
    #else
    TICK();
    if (std::is_same<scalar_type, double>::value) {
       status = cusparseDcsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               nrow, ncol, A.nnzU,
                               (const double*)&mone,  A.descrU,
                                                      (double*)A.d_Unzvals, A.d_Urow_ptr, A.d_Ucol_idx,
                                                      (double*)d_xv,
                               (const double*)&one,   (double*)d_bv);
    } else if (std::is_same<scalar_type, float>::value) {
       status = cusparseScsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               nrow, ncol, A.nnzU,
                               (const float*)&mone, A.descrA,
                                                    (float*)A.d_Unzvals, A.d_Urow_ptr, A.d_Ucol_idx,
                                                    (float*)d_xv,
                               (const float*)&one,  (float*)d_bv);
    }
    TOCK(x.time1);
    if (CUSPARSE_STATUS_SUCCESS != status) {
      printf( " Failed cusparseDcsrmv for GS\n" );
    }
    #endif

#elif defined(HPGMP_WITH_HIP)
  if (hipSuccess != hipMemcpy(d_bv, r.d_values(), nrow*sizeof(scalar_type), hipMemcpyDeviceToDevice)) {
    printf( " Failed to memcpy d_r\n" );
  }
  rocsparse_datatype rocsparse_compute_type = rocsparse_datatype_f64_r;
  if (std::is_same<scalar_type, float>::value) {
    rocsparse_compute_type = rocsparse_datatype_f32_r;
  }
  size_t buffer_size = A.buffer_size_U;
  rocsparse_dnvec_descr vecX, vecY;
  rocsparse_create_dnvec_descr(&vecX, ncol, (void*)d_xv, rocsparse_compute_type);
  rocsparse_create_dnvec_descr(&vecY, nrow, (void*)d_bv, rocsparse_compute_type);
  TICK();
  if (rocsparse_status_success !=
      rocsparse_spmv
          (A.rocsparseHandle, rocsparse_operation_none,
           &mone, A.descrU, vecX, &one, vecY,
           rocsparse_compute_type, rocsparse_spmv_alg_default,
           #if ROCM_VERSION >= 50400
	   rocsparse_spmv_stage_compute,
           #endif
           &buffer_size, A.buffer_U)) {
    printf( " Failed rocsparse_spmv\n" );
  }
  TOCK(x.time1);
#endif

  // x = L^{-1}b
  TICK();
#if defined(HPGMP_WITH_CUDA)
    if (std::is_same<scalar_type, double>::value) {
      #if CUDA_VERSION >= 11000
      status = cusparseDcsrsv2_solve(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, nrow, A.nnzL,
                                     (const double*)&one, A.descrL,
                                           (double*)A.d_Lnzvals, A.d_Lrow_ptr, A.d_Lcol_idx,
                                            A.infoL,
                                     (double*)d_bv, (double*)d_xv,
                                     CUSPARSE_SOLVE_POLICY_USE_LEVEL, A.buffer_L);
      #else
      status = cusparseDcsrsv_solve(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    nrow,
                                    (const double*)&one, A.descrL,
                                          (double*)A.d_Lnzvals, A.d_Lrow_ptr, A.d_Lcol_idx,
                                           A.infoL,
                                    (double*)d_bv, (double*)d_xv);
      #endif
    } else if (std::is_same<scalar_type, float>::value) {
      #if CUDA_VERSION >= 11000
      status = cusparseScsrsv2_solve(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, nrow, A.nnzL,
                                     (const float*)&one, A.descrL,
                                           (float*)A.d_Lnzvals, A.d_Lrow_ptr, A.d_Lcol_idx,
                                            A.infoL,
                                     (float*)d_bv, (float*)d_xv,
                                     CUSPARSE_SOLVE_POLICY_USE_LEVEL, A.buffer_L);
      #else
      status = cusparseScsrsv_solve(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    nrow,
                                    (const float*)&one, A.descrL,
                                          (float*)A.d_Lnzvals, A.d_Lrow_ptr, A.d_Lcol_idx,
                                                  A.infoL,
                                    (float*)d_bv, (float*)d_xv);
      #endif
    }
    if (CUSPARSE_STATUS_SUCCESS != status) {
      printf( " Failed cusparseDcsrsv_solve\n" );
      if (status == CUSPARSE_STATUS_NOT_INITIALIZED)            printf( " > CUSPARSE_STATUS_NOT_INITIALIZED <\n" );
      if (status == CUSPARSE_STATUS_ALLOC_FAILED)               printf( " > CUSPARSE_STATUS_ALLOC_FAILED <\n" );
      if (status == CUSPARSE_STATUS_INVALID_VALUE)              printf( " > CUSPARSE_STATUS_INVALID_VALUE <\n" );
      if (status == CUSPARSE_STATUS_EXECUTION_FAILED)           printf( " > CUSPARSE_STATUS_EXECUTION_FAILED <\n" );
      if (status == CUSPARSE_STATUS_INTERNAL_ERROR)             printf( " > CUSPARSE_STATUS_INTERNAL_ERROR <\n" );
      if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED ) printf( " > CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED <\n" );
    }
#elif defined(HPGMP_WITH_HIP)
  buffer_size = A.buffer_size_L;
  rocsparse_create_dnvec_descr(&vecX, nrow, (void*)d_bv, rocsparse_compute_type);
  rocsparse_create_dnvec_descr(&vecY, nrow, (void*)d_xv, rocsparse_compute_type);
  if (rocsparse_status_success != rocsparse_spsv(A.rocsparseHandle, rocsparse_operation_none,
                                                 &one, A.descrL, vecX, vecY, rocsparse_compute_type,
                                                 rocsparse_spsv_alg_default, rocsparse_spsv_stage_compute,
                                                 &buffer_size, A.buffer_L)) {
     printf( " Failed rocsparse_spsv(solve, nrow=%d, %s)\n",nrow,(rocsparse_compute_type == rocsparse_datatype_f32_r ? "single" : "double"));
  }
  #if 0 // TODO: copying output to host..
  if (hipSuccess != hipMemcpy(xv, d_xv, nrow*sizeof(scalar_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #endif
#endif
  TOCK(x.time2);

#ifdef HPGMP_DEBUG
  scalar_type * tv = (scalar_type *)malloc(nrow * sizeof(scalar_type));
  // copy x to host for check inside WAXPBY (debug)
  #if defined(HPGMP_WITH_CUDA)
  if (cudaSuccess != cudaMemcpy(tv, d_xv, nrow*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_b\n" );
  }
  #else
  if (hipSuccess != hipMemcpy(tv, d_xv, nrow*sizeof(scalar_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_b\n" );
  }
  #endif
  
  scalar_type l_enorm = 0.0;
  scalar_type l_xnorm = 0.0;
  scalar_type l_rnorm = 0.0;
  for (int j=0; j<nrow; j++) {
    l_xnorm += tv[j]*tv[j];
  }
  for (int j=0; j<nrow; j++) {
    l_enorm += (xv[j]-tv[j])*(xv[j]-tv[j]);
    l_rnorm += rv[j]*rv[j];
  }
  scalar_type enorm = 0.0;
  scalar_type xnorm = 0.0;
  scalar_type rnorm = 0.0;
  #ifndef HPGMP_NO_MPI
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();
  MPI_Allreduce(&l_enorm, &enorm, 1, MPI_SCALAR_TYPE, MPI_SUM, A.comm);
  MPI_Allreduce(&l_xnorm, &xnorm, 1, MPI_SCALAR_TYPE, MPI_SUM, A.comm);
  MPI_Allreduce(&l_rnorm, &rnorm, 1, MPI_SCALAR_TYPE, MPI_SUM, A.comm);
  #else
  enorm = l_enorm;
  xnorm = l_xnorm;
  rnorm = l_rnorm;
  #endif
  enorm = sqrt(enorm);
  xnorm = sqrt(xnorm);
  rnorm = sqrt(rnorm);
  int rank = 0;
  MPI_Comm_rank(A.comm, &rank);
  if (rank == 0) {
    HPGMP_fout << rank << " : GS_forward(" << nrow << " x " << ncol << "): error = " << enorm << " (x=" << xnorm << ", r=" << rnorm << ")" << std::endl;
  }
  free(tv);
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeGS_Forward_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&);

template
int ComputeGS_Forward_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&);

#endif

