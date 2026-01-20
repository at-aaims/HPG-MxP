
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

/*!
 @file Vector.hpp

 HPGMP routine
 */

#ifndef COMPUTE_TRSM_HPP
#define COMPUTE_TRSM_HPP

#include "Geometry.hpp"
#include "SerialDenseMatrix.hpp"

template<class SerialDenseMatrix_type>
int ComputeTRSM(const local_int_t n,
                const typename SerialDenseMatrix_type::scalar_type alpha,
                const SerialDenseMatrix_type& U,
                SerialDenseMatrix_type& x);

#endif // COMPUTE_TRSM_HPP
