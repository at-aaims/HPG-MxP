
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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "DataTypes.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
int OptimizeProblem(SparseMatrix_type& A, GMRESData_type& data, Vector_type& b, Vector_type& x,
                    Vector_type& xexact, const HPGMP_gen_opts& gopts);

/** @brief Helper function that reports memory usage of the optimization proces.
 *
 * It should be implemented in a non-trivial way if OptimizeProblem is non-trivial
 * This value will be used to report Gbytes used in ReportResults
 * (the value returned will be divided by 1000000000.0).
 *
 * @return As type double, the total number of bytes allocated during and retained
 *         after calling OptimizeProblem.
 */

template<class SparseMatrix_type>
double OptimizeProblemMemoryUse(const SparseMatrix_type& A, const HPGMP_gen_opts& gopts);

#endif // OPTIMIZEPROBLEM_HPP
