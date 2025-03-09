
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
 @file ComputeDotProduct_ref.cpp

 HPGMP routine
 */
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_OPENMP
 #include <omp.h>
#endif
#include <cassert>

#ifndef HPGMP_NO_MPI
 #include <mpi.h>
 #include "mytimer.hpp"
 #include "Utils_MPI.hpp"
#endif

#include "ComputeDotProduct_ref.hpp"
#include "hpgmp.hpp"
#include "DataTypes.hpp"


/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
template<class Vector_type, class output_scalar_type>
int ComputeDotProduct_ref(const local_int_t n, const Vector_type & x, const Vector_type & y,
                          output_scalar_type & result, double & time_allreduce) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  output_scalar_type local_result (0.0);

#if defined(HPGMP_DEBUG)
  using input_scalar_type = typename Vector_type::scalar_type;
  input_scalar_type * xv = x.values;
  input_scalar_type * yv = y.values;
  if (yv==xv) {
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
  } else {
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
  }
#endif

  using input_scalar_type = typename Vector_type::scalar_type; 
  input_scalar_type* d_x = x.d_values;
  input_scalar_type* d_y = y.d_values;

  #ifdef HPGMP_DEBUG
  output_scalar_type local_tmp = local_result;
  #endif
  #if defined(HPGMP_WITH_CUDA)
  // Compute dot on Nvidia GPU
  cublasHandle_t handle = x.handle;
  if (std::is_same<input_scalar_type, double>::value) {
    double double_result;
    if (CUBLAS_STATUS_SUCCESS != cublasDdot (handle, n, (double*)d_x, 1, (double*)d_y, 1, (double*)&double_result)) {
      printf( " Failed cublasDdot\n" );
    }
    local_result = double_result;
  } else if (std::is_same<input_scalar_type, float>::value) {
    float float_result;
    if (CUBLAS_STATUS_SUCCESS != cublasSdot (handle, n, (float*)d_x, 1,  (float*)d_y, 1,  (float*)&float_result)) {
      printf( " Failed cublasSdot\n" );
    }
    local_result = float_result;
  }
  #elif defined(HPGMP_WITH_HIP)
  // Compute dot on AMD GPU
  rocblas_handle handle = x.handle;
  if (std::is_same<input_scalar_type, double>::value) {
    double double_result;
    if (rocblas_status_success != rocblas_ddot (handle, n, (double*)d_x, 1, (double*)d_y, 1, (double*)&double_result)) {
      printf( " Failed rocblas_ddot\n" );
    }
    local_result = double_result;
  } else if (std::is_same<input_scalar_type, float>::value) {
    float float_result;
    if (rocblas_status_success != rocblas_sdot (handle, n, (float*)d_x, 1,  (float*)d_y, 1,  (float*)&float_result)) {
      printf( " Failed rocblas_sdot\n" );
    }
    local_result = float_result;
  }
  #endif

#ifndef HPGMP_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  int size; // Number of MPI processes
  MPI_Comm_size(x.comm, &size);
  double t0 = mytimer();
  if (size > 1) {
      MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<output_scalar_type>::getType ();
      MPI_Op MPI_SCALAR_SUM = MpiTypeTraits<output_scalar_type>::getSumOp ();
      output_scalar_type global_result (0.0);
      MPI_Allreduce(&local_result, &global_result, 1, MPI_SCALAR_TYPE, MPI_SCALAR_SUM, x.comm);
      result = global_result;
  } else {
      result = local_result;
  }
  time_allreduce += mytimer() - t0;

  #if defined(HPGMP_WITH_CUDA) & defined(HPGMP_DEBUG)
  output_scalar_type global_tmp (0.0);
  MPI_Allreduce(&local_tmp, &global_tmp, 1, MPI_SCALAR_TYPE, MPI_SUM, x.comm);
  int rank = 0;
  MPI_Comm_rank(x.comm, &rank);
  if (rank == 0) {
    HPGMP_fout << rank << " : DotProduct(" << n << "): error = " << global_tmp-global_result << " (dot=" << global_result << ")" << std::endl;
  }
  #endif
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeDotProduct_ref<Vector<double> >(int, Vector<double> const&, Vector<double> const&, double&, double&);

template
int ComputeDotProduct_ref<Vector<float> >(int, Vector<float> const&, Vector<float> const&, float&, double&);

#endif
