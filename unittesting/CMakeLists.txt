include_directories(${CMAKE_SOURCE_DIR}/src)

if(NOT HPGMP_BUILD_REFERENCE AND NOT HPGMP_ENABLE_GINKGO)
    add_library(basic_main basic_main.cpp)
    target_link_libraries(basic_main PUBLIC hpgmp hpgmp_device)

    add_executable(test_spmv test_spmv.cpp)
    target_link_libraries(test_spmv hpgmp hpgmp_device)
    add_test(NAME spmv COMMAND test_spmv)

    add_executable(test_multicoloring multicoloring.cpp)
    target_link_libraries(test_multicoloring PRIVATE basic_main hpgmp_device)
    add_test(NAME multicolor_reordering COMMAND test_multicoloring)

    add_executable(test_gs_multicolor gs_multicolor.cpp)
    target_link_libraries(test_gs_multicolor hpgmp hpgmp_device)
    add_test(NAME gs_ell_multicolor COMMAND test_gs_multicolor)

    add_executable(test_dot dot.cpp)
    target_link_libraries(test_dot hpgmp hpgmp_device)
    add_test(NAME dot COMMAND test_dot)
endif()

if(HPGMP_ENABLE_GINKGO)
    add_executable(test_ginkgo test_ginkgo.cpp)
    target_link_libraries(test_ginkgo hpgmp hpgmp_device)
    add_test(NAME ginkgo COMMAND test_ginkgo)
endif()
