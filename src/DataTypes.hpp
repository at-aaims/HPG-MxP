
#ifndef HPGMP_DATA_TYPES_HPP
#define HPGMP_DATA_TYPES_HPP

#ifdef HPGMP_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#elif defined(HPGMP_WITH_HIP)
#include <hip/hip_runtime_api.h>
#include <rocm-core/rocm_version.h>
#define ROCM_VERSION ROCM_VERSION_MAJOR * 10000 + ROCM_VERSION_MINOR * 100 + ROCM_VERSION_PATCH
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>
#else
#include <stddef.h>
#endif

#if 0 // TODO: Revisit half precision support
#if defined HPGMP_WITH_CUDA || defined HPGMP_WITH_HIP
//#   define HALF_ROUND_STYLE 1         // round-to-nearest
//#   define HALF_ROUND_TIES_TO_EVEN 1
//#   include <half/half.hpp>
//using half_float::half;               // This clashes with rocrand which uses __half.
#if defined HPGMP_WITH_HIP
#include <hip/hip_fp16.h>
#endif
// CUDA/HIP
using half = __half;
#else
// CPU
using half = _Float16;
#endif
#endif

#if defined HPGMP_WITH_CUDA || defined HPGMP_WITH_HIP
#define HPGMP_WITH_ACCELERATION 1
#endif

#endif
