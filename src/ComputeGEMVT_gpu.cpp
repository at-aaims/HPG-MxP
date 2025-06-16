
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
 @file ComputeGEMVT_gpu.cpp

 HPGMP routine for computing GEMV transpose (dot-products)
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "DataTypes.hpp"
#include "ComputeGEMVT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT_ref(const local_int_t m, const local_int_t n,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & y) {

  HPGMP_RANGE_PUSH(__FUNCTION__);

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename            Vector_type::scalar_type scalarX_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarY_type;

  assert(x.local_length() >= m); // Test vector lengths
  assert(y.n_rows() >= n);
  assert(y.n_cols() == 1);

  // Output serial dense vector 
  scalarY_type * const yv = y.values();

#if defined(HPGMP_DEBUG)
  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  // Input serial dense vector 
  const scalarA_type * const Av = A.values();
  const scalarX_type * const xv = x.values();

  // GEMV on HOST CPU
  if (beta == zero) {
    for (local_int_t i = 0; i < n; i++)
        yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < n; i++)
        yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[j] += Av[i + j*m] * xv[i];
    }
  } else {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++)
        yv[j] += alpha * Av[i + j*m] * xv[i];
    }
  }
#endif

  const scalarA_type * const d_Av = A.d_values();
  const scalarX_type * const d_xv = x.d_values();
  scalarY_type * const d_yv = y.d_values();
  auto handle = x.get_blas_handle();

  static_assert(std::is_same<scalarX_type, scalarY_type>::value, "! MxP GEMVT not supported!");
  double t0; TICK();
  #if defined(HPGMP_WITH_CUDA)
  // Perform GEMV on device
  if (std::is_same<scalarX_type, double>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasDgemv(handle, CUBLAS_OP_T,
                                             m, n,
                                             (double*)&alpha, (double*)d_Av, m,
                                                              (double*)d_xv, 1,
                                             (double*)&beta,  (double*)d_yv, 1)){
      printf( " Failed cublasDgemv\n" );
    }
  } else if (std::is_same<scalarX_type, float>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasSgemv(handle, CUBLAS_OP_T,
                                             m, n,
                                             (float*)&alpha, (float*)d_Av, m,
                                                             (float*)d_xv, 1,
                                             (float*)&beta,  (float*)d_yv, 1)){
      printf( " Failed cublasSgemv\n" );
    }
  }

  #elif defined(HPGMP_WITH_HIP)
  // Perform GEMV on device
  if (std::is_same<scalarX_type, double>::value) {
    if (rocblas_status_success != rocblas_dgemv(handle, rocblas_operation_transpose,
                                                m, n,
                                                (double*)&alpha, (double*)d_Av, m,
                                                                 (double*)d_xv, 1,
                                                (double*)&beta,  (double*)d_yv, 1)){
      printf( " Failed rocblas_dgemv\n" );
    }
  } else if (std::is_same<scalarX_type, float>::value) {
    if (rocblas_status_success != rocblas_sgemv(handle, rocblas_operation_transpose,
                                                m, n,
                                                (float*)&alpha, (float*)d_Av, m,
                                                                (float*)d_xv, 1,
                                                (float*)&beta,  (float*)d_yv, 1)){
      printf( " Failed rocblas_sgemv\n" );
    }
  }
  #endif
  
  TIME(y.time1);
  
  TICK();

  // Copy input serial dense vector to host
  y.update_host_mirror();

#ifndef HPGMP_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  int size; // Number of MPI processes
  MPI_Comm_size(A.get_comm(), &size);
  if (size > 1) {
      MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
      MPI_Op MPI_SCALAR_SUM = MpiTypeTraits<scalarY_type>::getSumOp ();
      MPI_Allreduce(MPI_IN_PLACE, yv, n, MPI_SCALAR_TYPE, MPI_SCALAR_SUM, A.get_comm());
  }
  TIME(y.time2);
#else
  y.time2 = 0.0;
#endif

  HPGMP_RANGE_POP(__FUNCTION__);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMVT_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, Vector<double> const&, double, SerialDenseMatrix<double> &);

template
int ComputeGEMVT_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, Vector<float> const&, float, SerialDenseMatrix<float> &);

#endif
