#
# HPGMP Benchmark CMake configuration
#
cmake_minimum_required( VERSION 3.10 FATAL_ERROR )

project( hpgmp
    VERSION 3.1
    LANGUAGES CXX
    DESCRIPTION "High Performance Conjugate Gradient Benchmark (HPGMP) " )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# Options
#
option(HPGMP_ENABLE_CONTIGUOUS_ARRAYS "Enable contiguous arrays for better cache pre-fetch" ON)
option(HPGMP_ENABLE_CUBIC_RADICAL_SEARCH "Enable faster search for optimal 3D process grid" ON)
option(HPGMP_ENABLE_DEBUG "Enable debug build" OFF)
option(HPGMP_ENABLE_DETAILED_DEBUG "Enable detailed debug build" OFF)
option(HPGMP_VERBOSE "Print more messages (low-cost)" OFF)
option(HPGMP_ENABLE_MPI "Enable MPI support" ON)
option(HPGMP_GPU_AWARE_MPI "Enable GPU-aware MPI support" OFF)
option(HPGMP_ONLY_BLOCKING_COMMS "Use only blocking communication routines for halo exchanges" OFF)
option(HPGMP_ENABLE_LONG_LONG "Enable use of 'long long' type for global indices" ON)
option(HPGMP_ENABLE_OPENMP "Enable OpenMP support" OFF)
option(HPGMP_ENABLE_CUDA "Enable CUDA support" OFF)
option(HPGMP_ENABLE_HIP "Enable HIP support" OFF)
option(HPGMP_ENABLE_TESTS "Enable unit tests" ON)
option(HPGMP_BUILD_REFERENCE "Build reference implementation without optimizations" OFF)
option(HPGMP_ENABLE_ASAN "Build address sanitizer" OFF)
option(CMAKE_DEBUG "Enable CMake script debugging" OFF)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
if(HPGMP_ENABLE_ASAN)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ggdb -fsanitize=address -fno-omit-frame-pointer")
endif()

if (HPGMP_ENABLE_HIP)
    enable_language(HIP)
    #message(STATUS "ROCm path is ${ROCM_PATH}")
    # CMake modules
    list(APPEND CMAKE_MODULE_PATH
         ${CMAKE_CURRENT_SOURCE_DIR}/cmake
         ${ROCM_PATH}/lib/cmake/hip
         ${ROCM_PATH}/hip/cmake)

    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

    # For HIP-related macros
    find_package(HIP REQUIRED)
    # For HIP runtime library
    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
    find_package(rocsparse REQUIRED)
    find_package(rocrand REQUIRED)
    find_package(rocprim REQUIRED)

    if(CMAKE_DEBUG)
        get_property(importTargetsAfterHip DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
        message(STATUS "Imported targets: ${importTargetsAfterHip}")
    endif()

    set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -O0 -ggdb")
    if(HPGMP_ENABLE_ASAN)
        set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELWITHDEBINFO} -ggdb -fsanitize=address -fno-omit-frame-pointer")
    endif()
elseif(HPGMP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

set(device_sources
    src/Vector.dev.cpp src/ComputeWAXPBY_opt.dev.cpp
)
if(HPGMP_REFERENCE)
    set(device_sources ${device_sources} src/ExchangeHalo_gpu.cpp)
else()
    set(device_sources ${device_sources}
        src/ell_matrix.cpp src/multicoloring.dev.cpp src/permute.dev.cpp
        src/ell_multicolor_gs.cpp src/restriction.cpp src/prolongation.cpp
    )
endif()

# Flag source files as device source files
if(HPGMP_ENABLE_HIP)
    foreach(i ${device_sources})
        set_source_files_properties(${i} PROPERTIES LANGUAGE HIP)
        set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    endforeach()
    #hip_add_library(hpgmp_device ${device_sources})
    add_library(hpgmp_device ${device_sources})
    target_link_libraries(hpgmp_device PRIVATE hip::device)
elseif(HPGMP_ENABLE_CUDA)
    foreach(i ${device_sources})
        set_source_files_properties(${i} PROPERTIES LANGUAGE CUDA)
    endforeach()
    add_library(hpgmp_device ${device_sources})
    set_target_properties(hpgmp_device PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(hpgmp_device PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
    add_library(hpgmp_device ${device_sources})
endif()


set(host_sources
    src/device_ctx.cpp src/MultiVector.cpp src/SparseMatrix.cpp
    src/GMRES.cpp src/GMRES_IR.cpp src/TestGMRES.cpp
    src/ComputeResidual.cpp src/GenerateGeometry.cpp
    src/GenerateNonsymProblem.cpp src/GenerateNonsymProblem_v1_ref.cpp src/CheckProblem.cpp
    src/OptimizeProblem.cpp src/ReadHpgmpDat.cpp src/ReportResults.cpp
    src/SetupHalo.cpp src/SetupHalo_ref.cpp
    src/SetupMatrix.cpp src/SetupProblem.cpp
    src/ValidGMRES.cpp src/BenchGMRES.cpp
    src/WriteProblem.cpp
    src/YAML_Doc.cpp src/YAML_Element.cpp
    src/ComputeSPMV.cpp 
    src/ComputeDotProduct.cpp src/ComputeDotProduct_ref.cpp src/ComputeDotProduct_gpu.cpp src/ComputeDotProduct_blas.cpp
    src/ComputeTRSM.cpp
    src/ComputeGEMV.cpp src/ComputeGEMV_ref.cpp src/ComputeGEMV_blas.cpp src/ComputeGEMV_gpu.cpp
    src/ComputeGEMVT.cpp src/ComputeGEMVT_ref.cpp src/ComputeGEMVT_blas.cpp src/ComputeGEMVT_gpu.cpp
    src/ComputeGEMMT.cpp src/ComputeGEMMT_ref.cpp src/ComputeGEMMT_gpu.cpp
    src/finalize.cpp src/init.cpp src/mytimer.cpp
    src/ComputeWAXPBY.cpp src/ComputeWAXPBY_ref.cpp src/ComputeWAXPBY_gpu.cpp
    src/ComputeMG.cpp
    src/GenerateNonsymCoarseProblem.cpp
    src/ComputeOptimalShapeXYZ.cpp src/MixedBaseCounter.cpp
    src/CheckAspectRatio.cpp src/OutputFile.cpp src/matrix_base.cpp
)

if(HPGMP_REFERENCE)
    set(host_sources ${host_sources}
        src/ExchangeHalo.cpp src/ExchangeHalo_ref.cpp
        src/ComputeSYMGS.cpp src/ComputeSYMGS_ref.cpp
        src/ComputeSPMV_ref.cpp src/ComputeSPMV_gpu.cpp
        src/ComputeGS_Forward.cpp src/ComputeGS_Forward_ref.cpp src/ComputeGS_Forward_gpu.cpp
        src/ComputeMG_ref.cpp
        src/ComputeProlongation_ref.cpp src/ComputeRestriction_ref.cpp
        src/ComputeProlongation_gpu.cpp src/ComputeRestriction_gpu.cpp
        src/optimize_problem_ref.cpp
    )
else()
    set(host_sources ${host_sources}
        src/optimize_problem_ell_rb.cpp
    )
endif()

add_library(hpgmp ${host_sources})

target_link_libraries(hpgmp_device PRIVATE hpgmp)

if(HPGMP_BUILD_REFERENCE)
    message(WARNING "Building reference mode without optimizations.")
    target_compile_definitions(hpgmp PRIVATE HPGMP_REFERENCE)
    target_compile_definitions(hpgmp_device PRIVATE HPGMP_REFERENCE)
endif()

if (HPGMP_ENABLE_CONTIGUOUS_ARRAYS)
    target_compile_definitions(hpgmp PUBLIC HPGMP_CONTIGUOUS_ARRAYS)
else()
    message(SEND_ERROR "Optimized HPGMP currently only works for contiguous sparse matrix arrays!")
endif ()

if (HPGMP_ENABLE_CUBIC_RADICAL_SEARCH)
    target_compile_definitions(hpgmp PUBLIC HPGMP_CUBIC_RADICAL_SEARCH)
endif ()

if (HPGMP_ENABLE_DEBUG)
    message(WARNING "HPGMP debug mode is enabled!")
    target_compile_definitions(hpgmp PUBLIC HPGMP_DEBUG)
endif ()
if (HPGMP_VERBOSE)
    target_compile_definitions(hpgmp PUBLIC HPGMP_VERBOSE)
endif ()

if (HPGMP_ENABLE_DETAILED_DEBUG)
    target_compile_definitions(hpgmp PUBLIC HPGMP_DETAILED_DEBUG)
endif ()

if (HPGMP_ENABLE_MPI)
    find_package(MPI REQUIRED)
    if (HPGMP_GPU_AWARE_MPI)
      string(SUBSTRING $ENV{PE_MPICH_GTL_DIR_amd_gfx90a} 2 -1 GTL_DIR)
      find_library(GTL_LIB mpi_gtl_hsa PATHS ${GTL_DIR})
      if(${GTL_LIB} STREQUAL "GTL_LIB-NOTFOUND")
        message(SEND_ERROR "MPI and GPU-aware MPI were requested but GTL was not found!")
      endif()
      message(STATUS "GTL library found: ${GTL_LIB}")
      target_compile_definitions(hpgmp PUBLIC HPGMP_USE_GPU_AWARE_MPI)
      target_compile_definitions(hpgmp_device PUBLIC HPGMP_USE_GPU_AWARE_MPI)
    endif()
    message(STATUS "MPI found. Include directories: ${MPI_CXX_INCLUDE_DIRS}")
    target_link_libraries(hpgmp PUBLIC MPI::MPI_CXX ${GTL_LIB})
    target_link_libraries(hpgmp_device PUBLIC MPI::MPI_CXX ${GTL_LIB})
    if(HPGMP_ONLY_BLOCKING_COMMS)
        message(WARNING "Compiling with blocking commications.")
        target_compile_definitions(hpgmp PRIVATE HPGMP_ONLY_BLOCKING_COMMS)
        target_compile_definitions(hpgmp_device PRIVATE HPGMP_ONLY_BLOCKING_COMMS)
    endif()
else ()
    target_compile_definitions(hpgmp PUBLIC HPGMP_NO_MPI)
    target_compile_definitions(hpgmp_device PUBLIC HPGMP_NO_MPI)
endif ()

if (NOT HPGMP_ENABLE_LONG_LONG)
    target_compile_definitions(hpgmp PUBLIC HPGMP_NO_LONG_LONG)
endif ()

if (HPGMP_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(hpgmp PUBLIC ${OpenMP_CXX_LIBRARIES})
else ()
    target_compile_definitions(hpgmp PUBLIC HPGMP_NO_OPENMP)
endif ()

if (HPGMP_ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(hpgmp PUBLIC HPGMP_WITH_CUDA)
    target_compile_definitions(hpgmp_device PUBLIC HPGMP_WITH_CUDA)
    target_compile_definitions(hpgmp PUBLIC HPGMP_USE_FENCE)
    target_link_libraries(hpgmp_device PUBLIC CUDA::cublas CUDA::cusparse CUDA::curand)
    target_link_libraries(hpgmp PUBLIC
                          CUDA::cublas CUDA::cusparse CUDA::curand)
else ()
    target_compile_definitions(hpgmp PUBLIC HPGMP_NO_CUDA)
endif ()

if (HPGMP_ENABLE_HIP)
    message(STATUS "HIP libraries are ${HIP_LIBRARIES}")
    target_compile_definitions(hpgmp PUBLIC HPGMP_WITH_HIP)
    target_compile_definitions(hpgmp_device PUBLIC HPGMP_WITH_HIP)
    target_compile_definitions(hpgmp PUBLIC HPGMP_USE_FENCE)
    target_link_libraries(hpgmp_device PRIVATE roc::rocprim)
    target_link_libraries(hpgmp PUBLIC hip::host roc::rocrand)
    target_link_libraries(hpgmp PUBLIC roc::rocblas roc::rocsparse)
else ()
    target_compile_definitions(hpgmp PUBLIC HPGMP_NO_HIP)
endif ()


if (HPGMP_ENABLE_BLAS)
    target_compile_definitions(hpgmp PUBLIC HPGMP_WITH_BLAS)
    message(WARNING "CPU BLAS is enabled")
endif ()

add_executable( xhpgmp src/main_hpgmp.cpp )
target_link_libraries(xhpgmp PRIVATE hpgmp hpgmp_device)

add_executable( xhpgmp_time src/main_time.cpp )
target_link_libraries(xhpgmp_time PRIVATE hpgmp hpgmp_device)
    
install(TARGETS xhpgmp xhpgmp_time)

if(HPGMP_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(unittesting)
endif()

