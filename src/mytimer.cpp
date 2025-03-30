
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

/////////////////////////////////////////////////////////////////////////

// Function to return time in seconds.
// If compiled with no flags, return CPU time (user and system).
// If compiled with -DWALL, returns elapsed time.

/////////////////////////////////////////////////////////////////////////

#include "mytimer.hpp"

#include <stdexcept>

#ifndef HPGMP_NO_MPI
#include <mpi.h>

double mytimer(void) {
  return MPI_Wtime();
}

#elif !defined(HPGMP_NO_OPENMP)

// If this routine is compiled with HPGMP_NO_MPI defined and not compiled with HPGMP_NO_OPENMP then use the OpenMP timer
#include <omp.h>
double mytimer(void) {
  return omp_get_wtime();
}
#else

#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>
double mytimer(void) {
  struct timeval tp;
  static long start=0, startu;
  if (!start) {
    gettimeofday(&tp, NULL);
    start = tp.tv_sec;
    startu = tp.tv_usec;
    return 0.0;
  }
  gettimeofday(&tp, NULL);
  return ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 ;
}

#endif

#if defined(HPGMP_USE_FENCE)
 #if defined(HPGMP_WITH_CUDA)
  //#include <cuda_runtime.h>
  //#include "cublas_v2.h"
void fence() {
  if(cudaSuccess != cudaDeviceSynchronize()) {
      throw std::runtime_error("Could synchronize CUDA device!");
  }
}

void fence(stream_t stream) {
    if(cudaSuccess != cudaStreamSynchronize(stream)) {
        throw std::runtime_error("Could synchronize CUDA stream!");
    }
}
 #elif defined(HPGMP_WITH_HIP)
  //#include "hip/hip_runtime.h"
void fence() {
  if(hipSuccess != hipDeviceSynchronize()) {
      throw std::runtime_error("Could synchronize HIP device!");
  }
}

void fence(stream_t stream) {
    if(hipSuccess != hipStreamSynchronize(stream)) {
        throw std::runtime_error("Could synchronize HIP stream!");
    }
}
 #endif
#else
void fence() {}
void fence(stream_t) {}
#endif


