
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

#ifndef GENERATE_NONSYM_COARSEPROBLEM_HPP
#define GENERATE_NONSYM_COARSEPROBLEM_HPP
#include "SparseMatrix.hpp"

template<class SparseMatrix_type>
void GenerateNonsymCoarseProblem(DeviceCtx *dctx, const SparseMatrix_type & A);

#endif // GENERATE_NONSYM_COARSEPROBLEM_HPP
