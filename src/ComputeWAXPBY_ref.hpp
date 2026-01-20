
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

#ifndef COMPUTEWAXPBY_REF_HPP
#define COMPUTEWAXPBY_REF_HPP
#include "Vector.hpp"

template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY_ref(const local_int_t n,
                      const typename VectorX_type::scalar_type alpha,
                      const VectorX_type& x,
                      const typename VectorY_type::scalar_type beta,
                      const VectorY_type& y,
                      VectorW_type& w);

#endif // COMPUTEWAXPBY_REF_HPP
