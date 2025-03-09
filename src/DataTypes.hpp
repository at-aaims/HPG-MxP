
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
#endif


#endif
