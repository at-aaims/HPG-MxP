
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTEDOTPRODUCT_REF_HPP
#define COMPUTEDOTPRODUCT_REF_HPP
#include "Vector.hpp"

template<class Vector_type, typename scalar_type = typename Vector_type::scalar_type>
int ComputeDotProduct_ref(const local_int_t n,
                          const Vector_type& x,
                          const Vector_type& y,
                          scalar_type& result,
                          double& time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP
