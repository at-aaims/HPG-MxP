
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

#if defined HPGMP_WITH_CUDA || defined HPGMP_WITH_HIP
//#   define HALF_ROUND_STYLE 1         // round-to-nearest
//#   define HALF_ROUND_TIES_TO_EVEN 1
//#   include <half/half.hpp>
//using half_float::half;               // This clashes with rocrand which uses __half.
// CUDA/HIP
#ifdef HPGMP_WITH_BFLOAT16
#if defined HPGMP_WITH_HIP
// Built-in type, since hip_bf16.h does not compile with a host-only compiler.
using half = __bf16;
#else
#include <cuda_bf16.h>
using half = __nv_bfloat16;
#endif
#else
#if defined HPGMP_WITH_HIP
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
using half = __half;
#endif
#else
// CPU
#ifdef HPGMP_WITH_BFLOAT16
using half = __bf16;
#else
using half = _Float16;
#endif
#endif

#if defined HPGMP_WITH_CUDA || defined HPGMP_WITH_HIP
#define HPGMP_WITH_ACCELERATION 1
#endif

/// Precision of the inner-solve vectors for a matrix stored in matrix_scalar_t.
template<typename matrix_scalar_t>
struct inner_vector_scalar
{
    using type = matrix_scalar_t;
};

// A half matrix uses single-precision vectors, unless HPGMP_WITH_HALF_VECTORS is defined.
#ifndef HPGMP_WITH_HALF_VECTORS
template<>
struct inner_vector_scalar<half>
{
    using type = float;
};
#endif

template<typename matrix_scalar_t>
using inner_vector_scalar_t = typename inner_vector_scalar<matrix_scalar_t>::type;

#endif
