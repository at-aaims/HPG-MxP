
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

#ifndef COMPUTEMG_REF_HPP
#define COMPUTEMG_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "perf_counter.hpp"

template<class SparseMatrix_type, class Vector_type>
int ComputeMG_ref(const SparseMatrix_type& A, const Vector_type& r, Vector_type& x, bool symmetric, perf_counters& ft);

#endif // COMPUTEMG_REF_HPP
