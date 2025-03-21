
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
 @file ComputeRestriction_ref.cpp

 HPGMP routine
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeRestriction_ref(const SparseMatrix_type & A, const Vector_type & rf) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;

  //scalar_type * Axfv = A.mgData->Axf->values;
  //scalar_type * rfv = rf.values;
  //scalar_type * rcv = A.mgData->rc->values;
  //local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->local_length();

  const scalar_type zero ( 0.0);
  const scalar_type  one ( 1.0);
  const scalar_type mone (-1.0);
  const local_int_t n = rf.local_length();
  scalar_type * d_Axfv = A.mgData->Axf->d_values();
  const scalar_type * d_rfv  = rf.d_values();
  scalar_type * d_rcv  = A.mgData->rc->d_values();
  #if defined(HPGMP_WITH_CUDA)
    cusparseStatus_t status;
    #if CUDA_VERSION >= 11000
    cudaDataType computeType;
    if (std::is_same<scalar_type, double>::value) {
      computeType = CUDA_R_64F;
    } else if (std::is_same<scalar_type, float>::value) {
      computeType = CUDA_R_32F;
    }
    // create matrix
    cusparseSpMatDescr_t Ac_cusparse;
    cusparseCreateCsr(&Ac_cusparse, nc, n, nc,
                      A.mgData->d_row_ptr, A.mgData->d_col_idx, A.mgData->d_nzvals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, computeType);
    // create vectors
    cusparseDnVecDescr_t vecAF, vecF, vecC;
    cusparseCreateDnVec(&vecAF, n,  d_Axfv, computeType);
    cusparseCreateDnVec(&vecF,  n,  d_rfv,  computeType);
    cusparseCreateDnVec(&vecC,  nc, d_rcv,  computeType);
    // SpMV
    status = cusparseSpMV(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &one, Ac_cusparse,
                                 vecF,
                          &zero, vecC,
                          computeType, CUSPARSE_MV_ALG_DEFAULT, A.mgData->buffer_R);
    if (CUSPARSE_STATUS_SUCCESS != status) {
      printf( " Failed first cusparseSpMV for Restriction\n" );
    } else {
      // SpMV
      status = cusparseSpMV(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &mone, Ac_cusparse,
                                   vecAF,
                            &one,  vecC,
                            computeType, CUSPARSE_MV_ALG_DEFAULT, A.mgData->buffer_R);
      if (CUSPARSE_STATUS_SUCCESS != status) {
        printf( " Failed second cusparseSpMV for Restriction\n" );
      }
    }
    #else
    if (std::is_same<scalar_type, double>::value) {
      if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    nc, n, nc,
                                                    (const double*)&one,  A.mgData->descrR,
                                                    (double*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                          d_rfv,
                                                    (const double*)&zero, (double*)d_rcv)) {
        printf( " Failed cusparseDcsrmv\n" );
      }
      if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    nc, n, nc,
                                                    (const double*)&mone, A.mgData->descrR,
                                                                          (double*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                          (double*)d_Axfv,
                                                    (const double*)&one,  (double*)d_rcv)) {
        printf( " Failed cusparseDcsrmv\n" );
      }
    } else if (std::is_same<scalar_type, float>::value) {
      if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    nc, n, nc,
                                                    (const float*)&one,  A.mgData->descrR,
                                                                         (float*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                         (float*)d_rfv,
                                                    (const float*)&zero, (float*)d_rcv)) {
        printf( " Failed cusparseScsrmv\n" );
      }
      if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    nc, n, nc,
                                                    (const float*)&mone, A.mgData->descrR,
                                                                         (float*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                         (float*)d_Axfv,
                                                    (const float*)&one,  (float*)d_rcv)) {
        printf( " Failed cusparseScsrmv\n" );
      }
    }
    #endif
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_datatype rocsparse_compute_type = rocsparse_datatype_f64_r;
  if (std::is_same<scalar_type, float>::value) {
    rocsparse_compute_type = rocsparse_datatype_f32_r;
  }
  size_t buffer_size = A.mgData->buffer_size_R;
  rocsparse_dnvec_descr vecX, vecY;
  rocsparse_create_dnvec_descr(&vecX, n,  (void*)d_rfv, rocsparse_compute_type);
  rocsparse_create_dnvec_descr(&vecY, nc, (void*)d_rcv, rocsparse_compute_type);
  if (rocsparse_status_success !=
      rocsparse_spmv
          (A.rocsparseHandle, rocsparse_operation_none,
           &one, A.mgData->descrR, vecX, &zero, vecY,
           rocsparse_compute_type, rocsparse_spmv_alg_default,
           #if ROCM_VERSION >= 50400
           rocsparse_spmv_stage_compute,
           #endif
           &buffer_size, A.mgData->buffer_R)) {
    printf( " Failed rocsparse_spmv\n" );
  }
  rocsparse_create_dnvec_descr(&vecX, n, (void*)d_Axfv, rocsparse_compute_type);
  if (rocsparse_status_success !=
      rocsparse_spmv
          (A.rocsparseHandle, rocsparse_operation_none,
           &mone, A.mgData->descrR, vecX, &one, vecY,
           rocsparse_compute_type, rocsparse_spmv_alg_default,
           #if ROCM_VERSION >= 50400
           rocsparse_spmv_stage_compute,
           #endif
           &buffer_size, A.mgData->buffer_R)) {
    printf( " Failed rocsparse_spmv\n" );
  }
  #endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeRestriction_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&);

template
int ComputeRestriction_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&);

#endif
