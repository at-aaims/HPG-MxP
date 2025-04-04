include_directories(${CMAKE_SOURCE_DIR}/src)

add_library(basic_main basic_main.cpp)
target_link_libraries(basic_main PUBLIC hpgmp hpgmp_device)

add_executable(test_spmv test_spmv.cpp)
target_link_libraries(test_spmv hpgmp hpgmp_device)
add_test(NAME ell_halo_setup COMMAND test_spmv)

add_executable(multicoloring multicoloring.cpp)
target_link_libraries(multicoloring PRIVATE basic_main)
add_test(NAME multicolor_reordering COMMAND multicoloring)

add_executable(test_gs_multicolor gs_multicolor.cpp)
target_link_libraries(test_gs_multicolor hpgmp hpgmp_device)
add_test(NAME gs_ell_multicolor COMMAND test_gs_multicolor)
