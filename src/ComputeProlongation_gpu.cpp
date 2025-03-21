
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
 @file ComputeProlongation_ref.cpp

 HPGMP routine
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeProlongation_ref(const SparseMatrix_type & Af, Vector_type & xf) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;

  //scalar_type * xfv = xf.values;
  //scalar_type * xcv = Af.mgData->xc->values;
  //local_int_t * f2c = Af.mgData->f2cOperator;
  const local_int_t nc = Af.mgData->rc->local_length();

  const scalar_type one (1.0);
  const local_int_t n = xf.local_length();
  scalar_type * d_xfv = xf.d_values();
  scalar_type * d_xcv = Af.mgData->xc->d_values();

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
    cusparseSpMatDescr_t Af_cusparse;
    cusparseCreateCsr(&Af_cusparse, nc, n, nc,
                      Af.mgData->d_row_ptr, Af.mgData->d_col_idx, Af.mgData->d_nzvals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, computeType);
    // create vectors
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, nc, d_xcv, computeType);
    cusparseCreateDnVec(&vecY, n,  d_xfv, computeType);
    // SpMV
    status = cusparseSpMV(Af.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                          &one, Af_cusparse,
                                vecX,
                          &one, vecY,
                          computeType, CUSPARSE_MV_ALG_DEFAULT, Af.mgData->buffer_P);
    if (CUSPARSE_STATUS_SUCCESS != status) {
      printf( " Failed cusparseSpMV for Prolongation\n" );
    }
    #else
    if (std::is_same<scalar_type, double>::value) {
      status = cusparseDcsrmv(Af.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                              nc, n, nc,
                              (const double*)&one, Af.mgData->descrR,
                              (double*)Af.mgData->d_nzvals, Af.mgData->d_row_ptr, Af.mgData->d_col_idx,
                              (double*)d_xcv,
                              (const double*)&one, (double*)d_xfv);
    } else if (std::is_same<scalar_type, float>::value) {
      status = cusparseScsrmv(Af.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                              nc, n, nc,
                              (const float*)&one, Af.mgData->descrR,
                              (float*)Af.mgData->d_nzvals, Af.mgData->d_row_ptr, Af.mgData->d_col_idx,
                              (float*)d_xcv,
                              (const float*)&one, (float*)d_xfv);
    }
    if (CUSPARSE_STATUS_SUCCESS != status) {
      printf( " Failed cusparseDcsrmv for Prolongation\n" );
    }
    #endif
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_datatype rocsparse_compute_type = rocsparse_datatype_f64_r;
  if (std::is_same<scalar_type, float>::value) {
    rocsparse_compute_type = rocsparse_datatype_f32_r;
  }
  size_t buffer_size = Af.mgData->buffer_size_P;
  rocsparse_dnvec_descr vecX, vecY;
  rocsparse_create_dnvec_descr(&vecX, nc, (void*)d_xcv, rocsparse_compute_type);
  rocsparse_create_dnvec_descr(&vecY, n,  (void*)d_xfv, rocsparse_compute_type);
  if (rocsparse_status_success != 
      #if ROCM_VERSION >= 50400
      rocsparse_spmv_ex
      #else
      rocsparse_spmv
      #endif
          (Af.rocsparseHandle, rocsparse_operation_none,
           &one, Af.mgData->descrP, vecX, &one, vecY,
           rocsparse_compute_type, rocsparse_spmv_alg_default,
           #if ROCM_VERSION >= 50400
           rocsparse_spmv_stage_compute,
           #endif
           &buffer_size, Af.mgData->buffer_P)) {
    printf( " Failed rocsparse_spmv(%dx%d)\n",nc,n );
  }
  #endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeProlongation_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double>&);

template
int ComputeProlongation_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float>&);

#endif
