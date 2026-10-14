/**
 * @file kernel_helpers.hpp.inc
 * @copyright (c) 2025, Oak Ridge National Laboratory.
 */

#ifdef HPGMP_WITH_HIP

/** Use Clang builtin to replace CUDA nontemporal load function.
 *
 * If the value at a memory location is known to be used just once,
 * it's better to avoid caching in L1 cache using this function.
 */
template<typename T>
__device__ __forceinline__ T __ldcg(const T* ptr)
{
    return __builtin_nontemporal_load(ptr);
}

/// Use Clang builtin to replace CUDA nontemporal store function.
template<typename T>
__device__ __forceinline__ void __stcg(T* ptr, const T value)
{
    __builtin_nontemporal_store(value, ptr);
}

/// Half overload, since the Clang builtin does not accept 16-bit floating-point types.
__device__ __forceinline__ void __stcg(half* ptr, const half value)
{
    __builtin_nontemporal_store(*reinterpret_cast<const unsigned short*>(&value), reinterpret_cast<unsigned short*>(ptr));
}

#ifdef HPGMP_WITH_BFLOAT16
/// Bfloat16 overload of __shfl_xor, which HIP only provides for fp16.
__device__ __forceinline__ half __shfl_xor(const half var, const int lane_mask)
{
    const int bits = __builtin_bit_cast(unsigned short, var);
    return __builtin_bit_cast(half, static_cast<unsigned short>(__shfl_xor(bits, lane_mask)));
}
#endif

#endif

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)

/// Half overload of fma.
__device__ __forceinline__ half fma(const half a, const half b, const half c)
{
#if defined(HPGMP_WITH_BFLOAT16) && defined(HPGMP_WITH_HIP)
    return static_cast<half>(fmaf(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)));
#else
    return __hfma(a, b, c);
#endif
}

#endif

#if 0

/* Uncached load of double value.
 * This does not cache the loaded value in L1 cache.
 *
 * NOTE: These functions are unnecessary - Cuda already provides
 * T __ldcg(const T* addr);
 * Similarly,
 * void __stcg(T* addr, T value);
 * can be used to store bypassing the L1 cache.
 * In both the above instructions, the value is still cached in L2.
 */
__device__ __inline__ double cuda_ld_gbl_cg(const double *addr) {
  double return_value;
  asm("ld.global.cg.f64 %0, [%1];" : "=d"(return_value) : "l"(addr));
  return return_value;
}

/* Uncached load of float value.
 */
__device__ __inline__ float cuda_ld_gbl_cg(const float *addr) {
  float return_value;
  asm("ld.global.cg.f32 %0, [%1];" : "=f"(return_value) : "l"(addr));
  return return_value;
}

/* Uncached load of int value.
 */
__device__ __inline__ int cuda_ld_gbl_cg(const int *addr) {
  int return_value;
  asm("ld.global.cg.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

#endif
