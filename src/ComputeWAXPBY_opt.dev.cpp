
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
 @file ComputeWAXPBY_ref.cpp

 HPGMP routine
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#include <cassert>
#include <iostream>
#ifndef HPGMP_NO_OPENMP
 #include <omp.h>
#endif

#if defined(HPGMP_DEBUG) & !defined(HPGMP_NO_MPI)
 #include <mpi.h>
 #include "Utils_MPI.hpp"
#endif

#include "ComputeWAXPBY_ref.hpp"
#include "hpgmp.hpp"
#include "DataTypes.hpp"

#include "Profiling.hpp"

template <typename x_type, typename y_type, typename w_type>
__global__
void waxpby(const local_int_t n, const x_type alpha, const x_type *const __restrict__ xv,
            const y_type beta, const y_type *const __restrict__ yv, w_type *const __restrict__ wv)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        wv[i] = static_cast<w_type>(alpha) * static_cast<w_type>(xv[i])
            + static_cast<w_type>(beta) * static_cast<w_type>(yv[i]);
    }
}

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY_opt(const local_int_t n,
                      const typename VectorX_type::scalar_type alpha,
                      const VectorX_type & x,
                      const typename VectorY_type::scalar_type beta,
                      const VectorY_type & y,
                            VectorW_type & w,
                      bool& isoptimized)
{

  HPGMP_RANGE_PUSH(__FUNCTION__);

  isoptimized = true;
  assert(x.local_length()>=n); // Test vector lengths
  assert(y.local_length()>=n);

  // quick return
  if (n <= 0) {
    HPGMP_RANGE_POP(__FUNCTION__);
    return 0;
  }

  typedef typename VectorX_type::scalar_type scalarX_type;
  typedef typename VectorY_type::scalar_type scalarY_type;
  typedef typename VectorW_type::scalar_type scalarW_type;

  const scalarX_type * const d_xv = x.d_values();
  const scalarY_type * const d_yv = y.d_values();
  scalarW_type * const d_wv = w.d_values();
  auto handle = w.get_blas_handle();

  // Only uniform-precision supported
  if ((std::is_same<scalarX_type, double>::value && std::is_same<scalarY_type, double>::value && std::is_same<scalarW_type, double>::value) ||
      (std::is_same<scalarX_type, float >::value && std::is_same<scalarY_type, float >::value && std::is_same<scalarW_type, float >::value)) {

#if defined(HPGMP_WITH_CUDA)
    // Compute axpy on Nvidia GPU
    // w = x (assuming y is not w)
    if (cudaSuccess != cudaMemcpy(d_wv, d_xv, n*sizeof(scalarW_type), cudaMemcpyDeviceToDevice)) {
      printf( " Failed to memcpy d_w\n" );
    }
    if (std::is_same<scalarX_type, double>::value) {
      // w = alpha*w
      if (CUBLAS_STATUS_SUCCESS != cublasDscal (handle, n, (const double*)&alpha, (double*)d_wv, 1)) {
        printf( " Failed cublasDscal\n" );
      }
      // w += alpha*x
      if (CUBLAS_STATUS_SUCCESS != cublasDaxpy (handle, n, (const double*)&beta, (double*)d_yv, 1, (double*)d_wv, 1)) {
        printf( " Failed cublasDdot\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      // w = beta*y
      if (CUBLAS_STATUS_SUCCESS != cublasSscal (handle, n, (const float*)&alpha, (float*)d_wv, 1)) {
        printf( " Failed cublasSscal\n" );
      }
      // w += alpha*x
      if (CUBLAS_STATUS_SUCCESS != cublasSaxpy (handle, n, (const float*)&beta, (float*) d_yv, 1, (float*) d_wv, 1)) {
        printf( " Failed cublasDdot\n" );
      }
    }
#elif defined(HPGMP_WITH_HIP)
    // Compute axpy on Nvidia GPU
    // w = x (assuming y is not w)
    if (hipSuccess != hipMemcpy(d_wv, d_xv, n*sizeof(scalarW_type), hipMemcpyDeviceToDevice)) {
      printf( " Failed to memcpy d_w\n" );
    }
    if (std::is_same<scalarX_type, double>::value) {
      // w = alpha*w
      if (rocblas_status_success != rocblas_dscal (handle, n, (const double*)&alpha, (double*)d_wv, 1)) {
        printf( " Failed rocblas_dscal\n" );
      }
      // w += alpha*x
      if (rocblas_status_success != rocblas_daxpy (handle, n, (const double*)&beta, (double*)d_yv, 1, (double*)d_wv, 1)) {
        printf( " Failed roocblas_ddot\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      // w = beta*y
      if (rocblas_status_success != rocblas_sscal (handle, n, (const float*)&alpha, (float*)d_wv, 1)) {
        printf( " Failed rocblas_sscal\n" );
      }
      // w += alpha*x
      if (rocblas_status_success != rocblas_saxpy (handle, n, (const float*)&beta, (float*) d_yv, 1, (float*) d_wv, 1)) {
        printf( " Failed rocblas_ddot\n" );
      }
    }
#endif

  } else {
      constexpr int threads_per_block = 1024;
      const int nblocks = (n - 1) / threads_per_block + 1;
      waxpby<<<nblocks, threads_per_block, 0, 0>>>(n, alpha, d_xv, beta, d_yv, d_wv);
  }

  HPGMP_RANGE_POP(__FUNCTION__);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeWAXPBY_opt< Vector<double>, Vector<double>, Vector<double> >(int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&, bool& opt);

template
int ComputeWAXPBY_opt< Vector<float>, Vector<float>, Vector<float> >(int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&, bool& opt);


// mixed
template
int ComputeWAXPBY_opt< Vector<double>, Vector<float>, Vector<double> >(int, double, Vector<double> const&, float, Vector<float> const&, Vector<double>&, bool& opt);

#endif
