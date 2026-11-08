# Checks arithmetic and conversions of the built-in host half type, used for CPU builds and HIP bfloat16.
# CUDA and HIP fp16 get their half type from headers.
if(NOT HPGMP_ENABLE_CUDA AND (HPGMP_ENABLE_BFLOAT16 OR NOT HPGMP_ENABLE_HIP))
    include(CheckCXXSourceRuns)
    if(HPGMP_ENABLE_BFLOAT16)
        set(HPGMP_HALF_HOST_TYPE __bf16)
        set(HPGMP_HALF_BITS_OF_0_5 0x3F00)
        set(HPGMP_HALF_BITS_OF_0_25 0x3E80)
    else()
        set(HPGMP_HALF_HOST_TYPE _Float16)
        set(HPGMP_HALF_BITS_OF_0_5 0x3800)
        set(HPGMP_HALF_BITS_OF_0_25 0x3400)
    endif()
    set(HPGMP_HALF_HOST_TEST
        "#include <cstring>
int main()
{
    volatile float f = 0.5f;
    volatile double d = 0.25;
    const ${HPGMP_HALF_HOST_TYPE} from_float = static_cast<${HPGMP_HALF_HOST_TYPE}>(f);
    const ${HPGMP_HALF_HOST_TYPE} from_double = static_cast<${HPGMP_HALF_HOST_TYPE}>(d);
    const ${HPGMP_HALF_HOST_TYPE} product = from_float * from_float;
    unsigned short float_bits = 0;
    unsigned short double_bits = 0;
    unsigned short product_bits = 0;
    std::memcpy(&float_bits, &from_float, 2);
    std::memcpy(&double_bits, &from_double, 2);
    std::memcpy(&product_bits, &product, 2);
    const float back = static_cast<float>(from_float);
    const bool correct = float_bits == ${HPGMP_HALF_BITS_OF_0_5} && double_bits == ${HPGMP_HALF_BITS_OF_0_25}
                         && product_bits == ${HPGMP_HALF_BITS_OF_0_25} && back == 0.5f;
    return correct ? 0 : 1;
}"
    )
    check_cxx_source_runs("${HPGMP_HALF_HOST_TEST}" HPGMP_HALF_HOST_WORKS_${HPGMP_HALF_HOST_TYPE})
    if(NOT HPGMP_HALF_HOST_WORKS_${HPGMP_HALF_HOST_TYPE} AND HPGMP_ENABLE_BFLOAT16)
        # Host conversions to __bf16 can need the GCC runtime.
        set(CMAKE_REQUIRED_LIBRARIES gcc_s)
        check_cxx_source_runs("${HPGMP_HALF_HOST_TEST}" HPGMP_HALF_HOST_WORKS_WITH_GCC_S)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(HPGMP_HALF_HOST_WORKS_WITH_GCC_S)
            target_link_libraries(hpgmp PUBLIC gcc_s)
        endif()
    endif()
    if(NOT HPGMP_HALF_HOST_WORKS_${HPGMP_HALF_HOST_TYPE} AND NOT (HPGMP_ENABLE_BFLOAT16
                                                                  AND HPGMP_HALF_HOST_WORKS_WITH_GCC_S))
        message(
            SEND_ERROR
                "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot compile, link or correctly run ${HPGMP_HALF_HOST_TYPE} on the host, which half precision needs!"
        )
    endif()
endif()
