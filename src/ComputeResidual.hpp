
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

#ifndef COMPUTERESIDUAL_HPP
#define COMPUTERESIDUAL_HPP
#include "Vector.hpp"

template<class Vector_type>
int ComputeResidual(const local_int_t n, const Vector_type& v1, const Vector_type& v2,
                    typename Vector_type::scalar_type& residual);

#endif // COMPUTERESIDUAL_HPP
