
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
 @file BenchGMRES.cpp

 HPGMP routine
 */

// Changelog
// None
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdexcept>
using std::endl;

#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"

#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"

#include "BenchGMRES.hpp"
#include "mytimer.hpp"
#include "ReportResults.hpp"

/// Record execution time of SpMV and MG kernels for reporting times
template<class TestGMRESDataType, class SparseMatrixType, class VectorType>
void test_mg_spmv(MPI_Comm comm, DeviceCtx *dctx, const Geometry *const geom,
                      const SparseMatrixType& A, TestGMRESDataType& test_data);

/*!
  Benchmark the optimized GMRES implementation

  @param[in]      argc      the "argc" parameter passed to the main() function
  @param[in]      argv      the "argv" parameter passed to the main() function
  @param[in]      comm      the communicator use to run benchmark
  @param[in] validation_failure  Whether validation had failed, for reporting.

  @param[inout]   test_data the data structure with the results of the test including pass/fail information

  @return Returns zero on success and a non-zero value otherwise.

  @see GMRES()
 */
template<class TestGMRESDataType, class scalar_type, class scalar_type2, class project_type>
int BenchGMRES(int argc, char **argv, comm_type comm, DeviceCtx *const dctx, int numberOfMgLevels,
               const bool verbose, const bool validation_failure, const HPGMP_gen_opts& gopts,
               TestGMRESDataType & test_data)
{
  HPGMP_RANGE_PUSH(__FUNCTION__);
  typedef Vector<scalar_type> Vector_type;
  typedef SparseMatrix<scalar_type> SparseMatrix_type;
  typedef GMRESData<scalar_type> GMRESData_type;

  typedef Vector<scalar_type2> Vector_type2;
  typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
  typedef GMRESData<scalar_type2, project_type> GMRESData_type2;

  const double benchmark_begin_time = mytimer();

  //////////////////////////////////////////////////////////
  // Setup problem
  Geometry * geom = new Geometry;

  SparseMatrix_type A;
  GMRESData_type data;

  SparseMatrix_type2 A_lo;
  GMRESData_type2 data_lo;

  Vector_type b, x;
  SetupProblem("bench_",argc, argv, comm, dctx, numberOfMgLevels, verbose, geom, A, data,
               A_lo, data_lo, b, x, test_data);
  if(geom->rank == 0) {
      std::cout << "BenchGMRES: Set up problem. Running time = " << test_data.runningTime
                << std::endl;
  }

  // Record execution time of reference SpMV and MG kernels for reporting times
  test_mg_spmv<TestGMRESDataType, SparseMatrix_type, Vector_type>(comm, dctx, geom, A, test_data);

  const double setup_done = mytimer();
  if(geom->rank == 0) {
      std::cout << " BenchGMRES: setup time = " << setup_done - benchmark_begin_time << "s"
                << std::endl;
  }
  // =====================================================================
  // Benchmark parameters
#ifdef HPGMP_WITH_PROFILING
  const int maxIters = 1; // Will perform at least restart_length
#else
  const int maxIters = 300;
#endif
  // Any official benchmark result must run at least this many seconds
  //double minOfficialTime = 1800;
  const double minOfficialTime = 120; // for testing..
  test_data.minOfficialTime = minOfficialTime;

  int niters = 0;
  scalar_type normr (0.0);
  scalar_type normr0 (0.0);
  const scalar_type tolerance = 0.0;
  const int restart_length = test_data.restart_length;
  const bool precond = true;
  test_data.maxNumIters = maxIters;
    
  constexpr int n_fl_ops = TestGMRESDataType::n_fl_ops;
  constexpr int n_timed_ops = TestGMRESDataType::n_timed_ops;

  // =====================================================================
  // Run optimized GMRES (here, we are calling GMRES_IR) for a fixed number of iterations
  // and record the obtained Gflop/s
  if(gopts.run_type == run_t::benchmark || gopts.run_type == run_t::benchmark_no_ref
          || gopts.run_type == run_t::standalone_mxp)
  {
    //warmup
    x.fill_zero();
    GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters,
             normr, normr0, precond, verbose, test_data);
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if(geom->rank == 0) {
        std::cout << "BenchGMRES: Completed warmup GMRES-IR run." << std::endl;
    }
#endif
    if (verbose && A.geom->rank==0) {
      HPGMP_fout << "Warm-up runs" << endl;
    }

#ifdef HPGMP_WITH_PROFILING
    const int numberOfGmresCalls = 1;
#else // HPGMP_WITH_PROFILING
    const int timing_calls = 10;
    double gmresir_run_time = 0;
    
    for (int i=0; i < timing_calls; ++i) {
      x.fill_zero();

      const double time_tic = mytimer();
      const int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x,
                                restart_length, maxIters, tolerance, niters, normr, normr0,
                                precond, verbose, test_data);
      gmresir_run_time += (mytimer() - time_tic);
      if(i == 0) {
        if (A.geom->rank==0) {
            std::cout << "BenchGMRES: Time taken by first timing solve = " << gmresir_run_time
                      << std::endl;
            std::cout << "BenchGMRES: Iterations taken by first timing solve = " << niters
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
    avg_run_time /= geom->size;

    // Get number of iterations to fill the required time
    const int numberOfGmresCalls = test_data.runningTime >= 0.0 ?
        ceil(test_data.runningTime / avg_run_time) :
        ceil(test_data.minOfficialTime / avg_run_time);
#endif // HPGMP_WITH_PROFILING
    if (A.geom->rank==0) {
        std::cout << "Number of benchmarking GMRES runs will be " << numberOfGmresCalls
                  << std::endl;
    }

    //
    // benchmark runs
    //
    double time_solve_total = 0.0;
    test_data.numOfMGCalls = 0;
    test_data.numOfSPCalls = 0;
    for (int i=0; i < n_fl_ops; i++)
        test_data.flops[i] = 0.0;
    for (int i=0; i < n_timed_ops; i++) {
        test_data.times[i] = 0.0;
        test_data.times_comp[i] = 0.0;
        test_data.times_comm[i] = 0.0;
    }
    test_data.ctrs_bench.reset();

    for (int i=0; i < numberOfGmresCalls; ++i) {
      x.fill_zero();

      const double time_tic = mytimer();
      const int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x,
                                restart_length, maxIters, tolerance, niters, normr, normr0,
                                precond, verbose, test_data);
      const double time_toc = (mytimer() - time_tic);
      time_solve_total += time_toc;

      if (i == 0) {
      }

      if (ierr > 1) {
          HPGMP_fout << "NaN Error in call to GMRES-IR: " << ierr << ".\n" << endl;
          if (A.geom->rank==0)
              std::cout << "NaN Error in call to GMRES-IR!\n" << endl;
          MPI_Abort(comm, -25);
      }

      if (A.geom->rank==0)
      {
          std::cout << "Call [" << i << " / " << numberOfGmresCalls
                    << "] Number of GMRES-IR Iterations ["
                    << niters <<"] Scaled Residual [" << normr/normr0 << "]" << std::endl;
        if(verbose) {
            HPGMP_fout << "Call [" << i << " / " << numberOfGmresCalls
                       << "] Number of GMRES-IR Iterations ["
                       << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
            HPGMP_fout << " Time        " << time_toc << endl;
            HPGMP_fout << " Time/itr    " << time_toc / niters << endl;
        }
      }
    }
    if (A.geom->rank==0) {
        std::cout << "BenchGMRES: Completed benchmarking runs." << std::endl;
    }
    if (verbose && A.geom->rank==0) {
      double flops = test_data.flops[0];
      HPGMP_fout << "  Accumulated Time " << time_solve_total << " seconds." << endl;
      HPGMP_fout << "  Final Gflop/s    " << flops/1000000000.0 << "/" << time_solve_total << " = "
                 << (flops/1000000000.0)/time_solve_total
                 << " (n = " << A.totalNumberOfRows << ")" << endl;
    }
    test_data.optTotalTime = time_solve_total;
    test_data.numOfCalls = numberOfGmresCalls;

    test_data.optNumOfMGCalls = test_data.numOfMGCalls;
    test_data.optNumOfSPCalls = test_data.numOfSPCalls;
    for (int i=0; i<n_fl_ops; i++)
        test_data.opt_flops[i] = test_data.flops[i];
    test_data.opt_flops[1] = test_data.ctrs_bench.mg_gs.get_total_flops() +
                             test_data.ctrs_bench.mg_rp.get_total_flops();
    test_data.opt_flops[0] += test_data.opt_flops[1];
    test_data.optTotalFlops = test_data.opt_flops[0];
    for (int i=0; i<n_timed_ops; i++)
        test_data.opt_times[i] = test_data.times[i];

    for (int i=0; i<n_timed_ops; i++)
        test_data.opt_times_comp[i] = test_data.times_comp[i];
    for (int i=0; i<n_timed_ops; i++)
        test_data.opt_times_comm[i] = test_data.times_comm[i];
  } else {
    // If mxp was not run
    test_data.optTotalFlops = 0.0;
    test_data.optTotalTime  = 0.0;
  }

  const double benchmark_done = mytimer();
  if(A.geom->rank == 0) {
      std::cout << " BenchGMRES: Main benchmark time = " << benchmark_done - setup_done << "s"
                << std::endl;
  }

  // =====================================================================
  // (Optional)
  // Run reference GMRES implementation for a fixed number of iterations
  // and record the obtained Gflop/s
  if (gopts.run_type == run_t::benchmark || gopts.run_type == run_t::standalone_ref) {
#ifdef HPGMP_WITH_PROFILING
    const int n_ref_calls = 1;
#else
    const int n_ref_calls = 10;
#endif
    x.fill_zero();
    //warmup
    GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond,
          verbose, test_data);

    double time_solve_total = 0.0;

    for (int i=0; i<n_fl_ops; i++) {
        test_data.flops[i] = 0.0;
    }
    for (int i=0; i<n_timed_ops; i++) {
        test_data.times[i] = 0.0;
        test_data.times_comp[i] = 0.0;
        test_data.times_comm[i] = 0.0;
    }
    test_data.ctrs_ref.reset();

    //benchmark runs
    for (int i=0; i< n_ref_calls; ++i) {
      x.fill_zero();

      const double time_tic = mytimer();
      const int ierr = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters,
                             normr, normr0, precond, verbose, test_data);
      const double time_toc = (mytimer() - time_tic);
      time_solve_total += time_toc;

      if (ierr == 2) {
          HPGMP_fout << "NaN Error in call to GMRES: " << ierr << ".\n" << endl;
          if (A.geom->rank==0)
              std::cout << "NaN Error in call to GMRES!" << std::endl;
      }
      if (A.geom->rank==0) {
        HPGMP_fout << "Call [" << i << " / " << n_ref_calls << "]";
        HPGMP_fout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual ["
                   << normr/normr0 << "]" << endl;
        HPGMP_fout << " Time        " << time_toc << endl;
        HPGMP_fout << " Time/itr    " << time_toc / niters << endl;
        std::cout << " BenchGMRES reference Call [" << i << " / " << n_ref_calls << "]";
        std::cout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual ["
                  << normr/normr0 << "]" << endl;
        std::cout << "     Time        " << time_toc << endl;
        std::cout << "     Time/itr    " << time_toc / niters << endl;
      }
    }
    if (verbose && A.geom->rank==0)
    {
      const double flops = test_data.flops[0];
      HPGMP_fout << "  Accumulated Time " << time_solve_total << " seconds." << endl;
      HPGMP_fout << "  Final Gflop/s    " << flops/1000000000.0 << "/" << time_solve_total
                 << " = " << (flops/1000000000.0)/time_solve_total 
                 << " (n = " << A.totalNumberOfRows << ")" << endl;
    }
    test_data.refTotalTime  = test_data.times[0];

    test_data.refNumOfMGCalls = test_data.numOfMGCalls;
    test_data.refNumOfSPCalls = test_data.numOfSPCalls;
    for (int i=0; i<n_fl_ops; i++)
        test_data.ref_flops[i] = test_data.flops[i];
    test_data.ref_flops[1] = test_data.ctrs_ref.mg_gs.get_total_flops() +
                             test_data.ctrs_ref.mg_rp.get_total_flops();
    test_data.ref_flops[0] += test_data.ref_flops[1];
    test_data.refTotalFlops = test_data.ref_flops[0];
    for (int i=0; i<n_timed_ops; i++)
        test_data.ref_times[i] = test_data.times[i];

    for (int i=0; i<n_timed_ops; i++)
        test_data.ref_times_comp[i] = test_data.times_comp[i];
    for (int i=0; i<n_timed_ops; i++)
        test_data.ref_times_comm[i] = test_data.times_comm[i];
  } else {
    test_data.refTotalFlops = 0.0;
    test_data.refTotalTime  = 0.0;
  }

#ifndef HPGMP_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  const double refrun_done = mytimer();
  if(A.geom->rank == 0) {
      std::cout << " BenchGMRES: Reference run time = " << refrun_done - benchmark_done << "s"
          << std::endl;
  }
    
  // Report results 
  ReportResults(A, numberOfMgLevels, test_data, validation_failure, gopts);

  const double reporting_done = mytimer();
  if(A.geom->rank == 0) {
      std::cout << " BenchGMRES: Report generation time = " << reporting_done - refrun_done << "s"
          << std::endl;
  }

  // cleanup
  DeleteMatrix(A);  
  DeleteMatrix(A_lo);

  if (verbose && A.geom->rank==0) {
    const auto total_benchmark_time = (mytimer() - benchmark_begin_time);
    HPGMP_fout << " Total benchmark time : " << total_benchmark_time << " seconds." << endl;
    std::cout << " Total benchmark time : " << total_benchmark_time << " seconds." << endl;
  }

  delete geom;
  HPGMP_RANGE_POP(__FUNCTION__);
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform version
template
int BenchGMRES< TestGMRESData<double>, double, double, double >
  (int, char**, comm_type, DeviceCtx*, int, bool, bool, const HPGMP_gen_opts&,
   TestGMRESData<double>&);

template
int BenchGMRES< TestGMRESData<float>, float, float, float >
  (int, char**, comm_type, DeviceCtx*, int, bool, bool, const HPGMP_gen_opts&,
   TestGMRESData<float>&);


// mixed version
template
int BenchGMRES< TestGMRESData<double>, double, float, float >
  (int, char**, comm_type, DeviceCtx*, int, bool, bool, const HPGMP_gen_opts&,
   TestGMRESData<double>&);


template<class TestGMRESDataType, class SparseMatrixType, class VectorType>
void test_mg_spmv(MPI_Comm comm, DeviceCtx *const dctx, const Geometry *const geom,
                  const SparseMatrixType& A, TestGMRESDataType& test_data)
{
    const local_int_t nrow = A.localNumberOfRows;
    const local_int_t ncol = A.localNumberOfColumns;
    const bool symmetric = false;

    VectorType x_overlap(ncol, A.comm, dctx),  // overlapped copy of x vector
               b_computed(nrow, A.comm, dctx);

    // load vector with random values
    x_overlap.fill_random();

    int ierr = 0;
    const int numberOfCalls = 10;
    const double t_begin = mytimer();
    double mgtime = 0, t0 = 0;
    perf_counters ft, ft0;
    for (int i=0; i< numberOfCalls; ++i) {
      ierr = ComputeSPMV(A, x_overlap, b_computed); // b_computed = A*x_overlap
#ifdef HPGMP_VERBOSE
      if(geom->rank == 0) {
          std::cout << "  test_mg_spmv: Completed SpMV." << std::endl;
      }
#endif
      if (ierr) HPGMP_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;

      if(i == 0) {
          ierr = ComputeMG(A, b_computed, x_overlap, symmetric, ft0); // b_computed = Minv*y_overlap
      } else {
          TICK();
          ierr = ComputeMG(A, b_computed, x_overlap, symmetric, ft); // b_computed = Minv*y_overlap
          TOCK(mgtime);
      }
      if (ierr) HPGMP_fout << "Error in call to MG: " << ierr << ".\n" << endl;
    }
#ifdef HPGMP_VERBOSE
    if(geom->rank == 0) {
        std::cout << "test_mg_spmv: Completed SpMV and MG refs." << std::endl;
    }
#endif
    // Total time divided by number of calls.
    test_data.SpmvMgTime = (mytimer() - t_begin)/((double) numberOfCalls);

    const double mgbytes = ft.mg_gs.get_total_memory_bytes() + ft.mg_rp.get_total_memory_bytes();
    const double mgflops = ft.mg_gs.get_total_flops() + ft.mg_rp.get_total_flops();
    if(geom->rank == 0) {
        std::cout << "BenchGMRES:  test_mg_spmv: MG = " << mgbytes / mgtime / 1e9
            << " GB/s" << std::endl;
        std::cout << "BenchGMRES:  test_mg_spmv: MG = " << mgflops / mgtime / 1e9
            << " GFLOP/s" << std::endl;
    }
}
