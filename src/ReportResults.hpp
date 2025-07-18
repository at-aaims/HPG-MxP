
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

#ifndef REPORTRESULTS_HPP
#define REPORTRESULTS_HPP
#include "SparseMatrix.hpp"
#include "TestGMRES.hpp"

template<class SparseMatrix_type, class TestGMRESData_type>
void ReportResults(const SparseMatrix_type & A, int numberOfMgLevels,
                   const TestGMRESData_type & testcg_data, int global_failure,
                   const HPGMP_gen_opts& gopts);

#endif // REPORTRESULTS_HPP
