
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

#ifndef GMRES_HPP
#define GMRES_HPP

#include "SparseMatrix.hpp"
#include "MultiVector.hpp"
#include "Vector.hpp"
#include "SerialDenseMatrix.hpp"
#include "GMRESData.hpp"

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
int GMRES(const SparseMatrix_type& A, GMRESData_type& data, const Vector_type& b, Vector_type& x,
          const int restart_length, const int max_iter, const typename SparseMatrix_type::scalar_type tolerance,
          int& niters, typename SparseMatrix_type::scalar_type& normr, typename SparseMatrix_type::scalar_type& normr0,
          bool doPreconditioning, bool verbose, TestGMRESData& test_data);

// this function will compute the Conjugate Gradient iterations.
// geom - Domain and processor topology information
// A - Matrix
// b - constant
// x - used for return value
// max_iter - how many times we iterate
// tolerance - Stopping tolerance for preconditioned iterations.
// niters - number of iterations performed
// normr - computed residual norm
// normr0 - Original residual
// times - array of timing information
// doPreconditioning - bool to specify whether or not symmetric GS will be applied.

#endif // GMRES_HPP
