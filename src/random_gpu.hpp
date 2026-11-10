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
// Aditya Kashi              (kashia@ornl.gov)
//
// ***************************************************
//@HEADER

#ifndef HPGMP_RANDOM_GPU_HPP
#define HPGMP_RANDOM_GPU_HPP

// Kept free of DataTypes.hpp, whose half type clashes with rocRAND's.
#include <cstddef>

#ifdef HPGMP_WITH_HIP
#include <hip/hip_runtime_api.h>
using random_stream_t = hipStream_t;
#elif defined HPGMP_WITH_CUDA
#include <cuda_runtime_api.h>
using random_stream_t = cudaStream_t;
#endif

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
/// Fills a device array with uniform random values using rocRAND or cuRAND.
void fill_random_gpu(double* x, size_t n, random_stream_t stream);
void fill_random_gpu(float* x, size_t n, random_stream_t stream);
#endif

#endif
