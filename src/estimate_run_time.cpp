#include "estimate_run_time.hpp"

#include <iostream>

#include "mytimer.hpp"
#include "GMRES_IR.hpp"

template<typename scalar_type, typename scalar_type2>
double estimate_run_time(comm_type comm,
                         const SparseMatrix<scalar_type>& A, const SparseMatrix<scalar_type2>& A_lo,
                         GMRESData<scalar_type>& data, GMRESData<scalar_type2>& data_lo,
                         const Vector<scalar_type>& b, Vector<scalar_type>& x, const int max_iters,
                         const int restart_length, const bool verbose)
{
    HPGMP_RANGE_PUSH(__FUNCTION__);

    TestGMRESData test_data{};

    int niters = 0;
    scalar_type normr(0.0);
    scalar_type normr0(0.0);
    const scalar_type tolerance = 0.0;
    const bool precond          = true;

    x.fill_zero();
    GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, max_iters, tolerance, niters,
             normr, normr0, precond, verbose, test_data);
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if (A.geom->rank == 0) {
        std::cout << "estimate_run_time: Completed warmup GMRES-IR run." << std::endl;
    }
#endif
    if (verbose && A.geom->rank == 0) {
        HPGMP_fout << "Warm-up runs" << std::endl;
    }

    const int timing_calls  = 10;
    double gmresir_run_time = 0;

    for (int i = 0; i < timing_calls; ++i) {
        x.fill_zero();

        const double time_tic = mytimer();
        const int ierr        = GMRES_IR(A, A_lo, data, data_lo, b, x,
                                         restart_length, max_iters, tolerance, niters, normr, normr0,
                                         precond, verbose, test_data);
        gmresir_run_time += (mytimer() - time_tic);
        if (i == 0) {
            if (A.geom->rank == 0) {
                std::cout << "estimate_run_time: Time taken by first timing solve = "
                          << gmresir_run_time << std::endl;
                std::cout << "estimate_run_time: Iterations taken by first timing solve = " << niters
                          << std::endl;
            }
        }
    }

    gmresir_run_time /= timing_calls;

    double avg_run_time = 0;
#ifndef HPGMP_NO_MPI
    MPI_Allreduce(&gmresir_run_time, &avg_run_time, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
    avg_run_time = gmresir_run_time;
#endif
    avg_run_time /= A.geom->size;

    HPGMP_RANGE_POP(__FUNCTION__);

    return avg_run_time;
}

template double estimate_run_time(comm_type comm,
                                  const SparseMatrix<double>& A, const SparseMatrix<float>& A_lo,
                                  GMRESData<double>& data, GMRESData<float>& data_lo,
                                  const Vector<double>& b, Vector<double>& x, int max_iters,
                                  int restart_length, bool verbose);

template double estimate_run_time(comm_type comm,
                                  const SparseMatrix<float>& A, const SparseMatrix<float>& A_lo,
                                  GMRESData<float>& data, GMRESData<float>& data_lo,
                                  const Vector<float>& b, Vector<float>& x, int max_iters,
                                  int restart_length, bool verbose);

template double estimate_run_time(comm_type comm,
                                  const SparseMatrix<double>& A, const SparseMatrix<double>& A_lo,
                                  GMRESData<double>& data, GMRESData<double>& data_lo,
                                  const Vector<double>& b, Vector<double>& x, int max_iters,
                                  int restart_length, bool verbose);
