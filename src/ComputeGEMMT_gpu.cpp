
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
 @file ComputeGEMMT_gpu.cpp

 HPGMP routine for computing GEMM transpose (dot-products)
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "DataTypes.hpp"
#include "ComputeGEMMT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"

template<class MultiVector_type, class SerialDenseMatrix_type>
int ComputeGEMMT_ref(const local_int_t m, const local_int_t n, const local_int_t k,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const MultiVector_type & B,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & C) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarC_type;

  // Output serial dense vector 
  scalarC_type * const Cv = C.values();

  const scalarA_type * const d_Av = A.d_values();
  const scalarA_type * const d_Bv = B.d_values();
  scalarC_type * const d_Cv = C.d_values();
  auto handle = A.get_blas_handle();

  double t0; TICK();
  #if defined(HPGMP_WITH_CUDA)
  // Perform GEMM on device
  if (std::is_same<scalarC_type, double>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                             m, n, k,
                                             (double*)&alpha, (double*)d_Av, k,
                                                              (double*)d_Bv, k,
                                             (double*)&beta,  (double*)d_Cv, m)){
      printf( " Failed cublasDgemv\n" );
    }
  } else if (std::is_same<scalarC_type, float>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                             m, n, k,
                                             (float*)&alpha, (float*)d_Av, k,
                                                             (float*)d_Bv, k,
                                             (float*)&beta,  (float*)d_Cv, m)){
      printf( " Failed cublasSgemv\n" );
    }
  }
  TIME(C.time1);

  #elif defined(HPGMP_WITH_HIP)
  // Perform GEMM on device
  if (std::is_same<scalarC_type, double>::value) {
    if (rocblas_status_success != rocblas_dgemm(handle,
                                                rocblas_operation_transpose,
                                                rocblas_operation_none,
                                                m, n, k,
                                                (double*)&alpha, (double*)d_Av, k,
                                                                 (double*)d_Bv, k,
                                                (double*)&beta,  (double*)d_Cv, m)){
      printf( " Failed rocblas_dgemv\n" );
    }
  } else if (std::is_same<scalarC_type, float>::value) {
    if (rocblas_status_success != rocblas_sgemm(handle,
                                                rocblas_operation_transpose,
                                                rocblas_operation_none,
                                                m, n, k,
                                                (float*)&alpha, (float*)d_Av, k,
                                                                (float*)d_Bv, k,
                                                (float*)&beta,  (float*)d_Cv, m)){
      printf( " Failed rocblas_sgemv\n" );
    }
  }
  TIME(C.time1);
  #endif
  
  TICK();
  // Copy output serial dense vector to host
  C.update_host_mirror();

#ifndef HPGMP_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  int size; // Number of MPI processes
  MPI_Comm_size(A.get_comm(), &size);
  if (size > 1) {
      MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarC_type>::getType ();
      MPI_Op MPI_SCALAR_SUM = MpiTypeTraits<scalarC_type>::getSumOp ();
      MPI_Allreduce(MPI_IN_PLACE, Cv, m*n, MPI_SCALAR_TYPE, MPI_SCALAR_SUM, A.get_comm());
  }
  TIME(C.time2);
#else
  C.time2 = 0.0;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMMT_ref< MultiVector<double>, SerialDenseMatrix<double> >
  (int, int, int, double, MultiVector<double> const&, MultiVector<double> const&, double, SerialDenseMatrix<double> &);

template
int ComputeGEMMT_ref< MultiVector<float>, SerialDenseMatrix<float> >
  (int, int, int, float, MultiVector<float> const&, MultiVector<float> const&, float, SerialDenseMatrix<float> &);

#endif
