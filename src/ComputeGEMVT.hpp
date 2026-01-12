
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
 @file ComputeGEMVT.hpp

 HPGMP routine
 */

#ifndef COMPUTE_GEMVT_HPP
#define COMPUTE_GEMVT_HPP

#include "Geometry.hpp"
#include "MultiVector.hpp"
#include "Vector.hpp"
#include "SerialDenseMatrix.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT(const local_int_t m, const local_int_t n,
                 const typename MultiVector_type::scalar_type alpha, const MultiVector_type& A, const Vector_type& x,
                 const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type& y,
                 bool& isOptimized);

#endif // COMPUTE_GEMVT
