#pragma once

#ifdef HPGMP_WITH_PROFILING

#if defined HPGMP_WITH_HIP || defined HPGMP_WITH_CUDA

#ifdef HPGMP_WITH_HIP
#include <roctracer/roctx.h>
#define HPGMP_RANGE_PUSH(x) roctxRangePush(x)
#define HPGMP_RANGE_POP(x)  roctxRangePop(); \
	                    roctxMarkA(x)
#endif // HPGMP_WITH_HIP

#ifdef HPGMP_WITH_CUDA
#include <nvToolsExt.h>
#define HPGMP_RANGE_PUSH(x) nvtxRangePush(x)
#define HPGMP_RANGE_POP(x)  nvtxRangePop(); \
	                    nvtxMarkA(x)
#endif // HPGMP_WITH_CUDA

#else

// Not using GPU
#define HPGMP_RANGE_PUSH(x)
#define HPGMP_RANGE_POP(x)

#endif // HPGMP_WITH_HIP || HPGMP_WITH_CUDA

#else

// Not using profiling
#define HPGMP_RANGE_PUSH(x)
#define HPGMP_RANGE_POP(x)

#endif // HPGMP_WITH_PROFILING
