include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test_spmv test_spmv.cpp)
target_link_libraries(test_spmv hpgmp hpgmp_device)

add_test(NAME ell_halo_setup
    COMMAND test_spmv)
