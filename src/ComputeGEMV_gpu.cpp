
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
 @file ComputeGEMV_gpu.cpp

 HPGMP routine for computing GEMV (vector-update)
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#include "ComputeGEMV_ref.hpp"
#include "hpgmp.hpp"
#include "DataTypes.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMV_ref(const local_int_t m, const local_int_t n,
                    const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const SerialDenseMatrix_type & x,
                    const typename      Vector_type::scalar_type beta,  const Vector_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  const scalarY_type one  (1.0);
  const scalarY_type zero (0.0);

  assert(x.m >= n); // Test vector lengths
  assert(x.n == 1);
  assert(y.localLength >= m);
    
  // Input serial dense vector 
  scalarX_type * const xv = x.values;

#if defined(HPGMP_DEBUG)
  scalarA_type * const Av = A.values;
  scalarY_type * const yv = y.values;
  // GEMV on HOST CPU
  if (beta == zero) {
    for (local_int_t i = 0; i < m; i++) yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[i] += Av[i + j*m] * xv[j];
    }
  } else {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[i] += alpha * Av[i + j*m] * xv[j];
    }
  }
#endif

  scalarA_type * const d_Av = A.d_values;
  scalarX_type * const d_xv = x.d_values;
  scalarY_type * const d_yv = y.d_values;
  if ((std::is_same<scalarX_type, double>::value && std::is_same<scalarY_type, double>::value && std::is_same<scalarA_type, double>::value) ||
      (std::is_same<scalarX_type, float >::value && std::is_same<scalarY_type, float >::value && std::is_same<scalarA_type, float >::value)) {

    #if defined(HPGMP_WITH_CUDA)
    // Copy input serial dense vector to device
    if (cudaSuccess != cudaMemcpy(d_xv, xv, n*sizeof(scalarX_type), cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_x\n" );
    }

    // Perform GEMV on device
    if (std::is_same<scalarX_type, double>::value) {
      if (CUBLAS_STATUS_SUCCESS != cublasDgemv(y.handle, CUBLAS_OP_N,
                                               m, n,
                                               (double*)&alpha, (double*)d_Av, m,
                                                                (double*)d_xv, 1,
                                               (double*)&beta,  (double*)d_yv, 1)){
        printf( " Failed cublasDgemv\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      if (CUBLAS_STATUS_SUCCESS != cublasSgemv(y.handle, CUBLAS_OP_N,
                                               m, n,
                                               (float*)&alpha, (float*)d_Av, m,
                                                               (float*)d_xv, 1,
                                               (float*)&beta,  (float*)d_yv, 1)){
        printf( " Failed cublasSgemv\n" );
      }
    }
    #elif defined(HPGMP_WITH_HIP)
    // Copy input serial dense vector to device
    if (hipSuccess != hipMemcpy(d_xv, xv, n*sizeof(scalarX_type), hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_x\n" );
    }

    // Perform GEMV on device
    if (std::is_same<scalarX_type, double>::value) {
      if (rocblas_status_success != rocblas_dgemv(y.handle, rocblas_operation_none,
                                                  m, n,
                                                  (double*)&alpha, (double*)d_Av, m,
                                                                   (double*)d_xv, 1,
                                                  (double*)&beta,  (double*)d_yv, 1)){
        printf( " Failed rocblas_dgemv\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      if (rocblas_status_success != rocblas_sgemv(y.handle, rocblas_operation_none,
                                                  m, n,
                                                  (float*)&alpha, (float*)d_Av, m,
                                                                  (float*)d_xv, 1,
                                                  (float*)&beta,  (float*)d_yv, 1)){
        printf( " Failed rocblas_sgemv\n" );
      }
    }
    #endif
  } else {
    HPGMP_vout << " Mixed-precision GEMV not supported" << std::endl;

    scalarA_type * const Av = A.values;
    scalarY_type * const yv = y.values;

    // Copy input matrix A from HOST CPU
    #if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(Av, d_Av, m*n*sizeof(scalarA_type), cudaMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    if (beta != zero) {
      if (cudaSuccess != cudaMemcpy(yv, d_yv, m*sizeof(scalarY_type), cudaMemcpyDeviceToHost)) {
        printf( " Failed to memcpy d_y\n" );
      }
    }
    #elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(Av, d_Av, m*n*sizeof(scalarA_type), hipMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    if (beta != zero) {
      if (hipSuccess != hipMemcpy(yv, d_yv, m*sizeof(scalarY_type), hipMemcpyDeviceToHost)) {
        printf( " Failed to memcpy d_y\n" );
      }
    }
    #endif

    // GEMV on HOST CPU
    if (beta == zero) {
      for (local_int_t i = 0; i < m; i++) yv[i] = zero;
    } else if (beta != one) {
      for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
    }

    if (alpha == one) {
      for (local_int_t i=0; i<m; i++) {
        for (local_int_t j=0; j<n; j++)
          yv[i] += Av[i + j*m] * xv[j];
      }
    } else {
      for (local_int_t i=0; i<m; i++) {
        for (local_int_t j=0; j<n; j++)
          yv[i] += alpha * Av[i + j*m] * xv[j];
      }
    }

    // Copy output vector Y from HOST CPU
    #if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(d_yv, yv, m*sizeof(scalarY_type), cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(d_yv, yv, m*sizeof(scalarY_type), hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #endif
  }

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMV_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, SerialDenseMatrix<double> const&, double, Vector<double> const&);

template
int ComputeGEMV_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, float, Vector<float> const&);


// mixed
template
int ComputeGEMV_ref< MultiVector<float>, Vector<double>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&);

#endif
