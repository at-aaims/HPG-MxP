#ifndef HPGMP_ESTIMATE_NUM_ITERS_HPP
#define HPGMP_ESTIMATE_NUM_ITERS_HPP

#include "SparseMatrix.hpp"
#include "GMRESData.hpp"

/**
 * Runs GMRES-IR with fixed iteration count a few times to measure
 * the average time taken per solve.
 */
#ifdef HPGMP_WITH_GINKGO // TODO: Improve this implementation
template<typename scalar_type, typename scalar_type2, class GMRESData_type, class GMRESData_type2>
double estimate_run_time(comm_type comm,
                         const SparseMatrix<scalar_type, scalar_type>& A, const SparseMatrix<scalar_type, scalar_type2>& A_lo,
                         GMRESData_type& data, GMRESData_type2& data_lo,
                         const Vector<scalar_type>& b, Vector<scalar_type>& x, int max_iters,
                         int restart_length, bool verbose);
#else
template<typename scalar_type, typename scalar_type2, class GMRESData_type, class GMRESData_type2>
double estimate_run_time(comm_type comm,
                         const SparseMatrix<scalar_type, scalar_type>& A, const SparseMatrix<scalar_type2, scalar_type2>& A_lo,
                         GGMRESData_type& data, GMRESData_type2& data_lo,
                         const Vector<scalar_type>& b, Vector<scalar_type>& x, int max_iters,
                         int restart_length, bool verbose);
#endif

#endif
