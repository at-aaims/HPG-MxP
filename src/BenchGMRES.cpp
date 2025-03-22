
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

#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"

#include "BenchGMRES.hpp"
#include "mytimer.hpp"
#include "ReportResults.hpp"

/// Record execution time of reference SpMV and MG kernels for reporting times
template<class TestGMRESDataType, class SparseMatrixType, class VectorType>
void test_mg_spmv_ref(MPI_Comm comm, DeviceCtx *dctx, const Geometry *const geom,
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
               bool verbose, bool runReference, const bool validation_failure,
               TestGMRESDataType & test_data)
{
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
  SetupProblem("bench_",argc, argv, comm, dctx, numberOfMgLevels, verbose, geom, A, data, A_lo, data_lo, b, x, test_data);
#ifdef HPGMP_VERBOSE
  MPI_Barrier(comm);
  if(geom->rank == 0) {
      std::cout << "BenchGMRES: Set up problem. Running time = " << test_data.runningTime
          << std::endl;
  }
#endif

  // Record execution time of reference SpMV and MG kernels for reporting times
  test_mg_spmv_ref<TestGMRESDataType, SparseMatrix_type, Vector_type>(comm, dctx, geom, A, test_data);

  // =====================================================================
  // Benchmark parameters
  int numberOfGmresCalls = 10;
  const int maxIters = 300;
  //double minOfficialTime = 1800; // Any official benchmark result must run at least this many seconds
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

  //const int num_flops = 4;
  //const int num_times = 12;
  //test_data.flops = (double*)malloc(num_flops * sizeof(double));
  //test_data.times = (double*)malloc(num_times * sizeof(double));
  //test_data.times_comp = (double*)malloc(num_times * sizeof(double));
  //test_data.times_comm = (double*)malloc(num_times * sizeof(double));

  // =====================================================================
  // Run optimized GMRES (here, we are calling GMRES_IR) for a fixed number of iterations
  // and record the obtained Gflop/s
  double time_solve_total = 0.0;
  {
    //warmup
    x.fill_zero();
    GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if(geom->rank == 0) {
        std::cout << "BenchGMRES: Completed warmup GMRES-IR run." << std::endl;
    }
#endif
    if (verbose && A.geom->rank==0) {
      HPGMP_fout << "Warm-up runs" << endl;
    }

    //benchmark runs
    test_data.numOfMGCalls = 0;
    test_data.numOfSPCalls = 0;
    for (int i=0; i < n_fl_ops; i++)
        test_data.flops[i] = 0.0;
    for (int i=0; i < n_timed_ops; i++) {
        test_data.times[i] = 0.0;
        test_data.times_comp[i] = 0.0;
        test_data.times_comm[i] = 0.0;
    }
    for (int i=0; i< numberOfGmresCalls; ++i) {
      x.fill_zero();

      const double time_tic = mytimer();
      const int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x,
                          restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
      const double time_toc = (mytimer() - time_tic);
      time_solve_total += time_toc;

      if (i == 0) {
        if (A.geom->rank==0) {
            std::cout << "BenchGMRES: Time taken by first solve = " << time_toc << std::endl;
            std::cout << "BenchGMRES: Iterations taken by first solve = " << niters << std::endl;
        }
        // Get correct number of iterations
        const int numberOfGmresCalls_min = test_data.runningTime >= 0.0 ?
            ceil(test_data.runningTime / time_toc) : ceil(test_data.minOfficialTime / time_toc);
        if (numberOfGmresCalls_min > numberOfGmresCalls) {
            numberOfGmresCalls = numberOfGmresCalls_min;
        }

        if (test_data.runningTime >= 0.0) {
            if (A.geom->rank==0) {
              if(verbose) {
                  HPGMP_fout << " numberOfGmresCalls = runningTime / time_toc = "
                            << test_data.runningTime << " / " << time_toc << " = " << numberOfGmresCalls << endl;
              }
              std::cout << "BenchGMRES: numberOfGmresCalls = runningTime / time_toc = "
                        << test_data.runningTime << " / " << time_toc << " = " << numberOfGmresCalls << endl;
            }
        } else {
            if (verbose && A.geom->rank==0) {
              HPGMP_fout << " numberOfGmresCalls = minOfficialTime / time_toc = "
                      << test_data.minOfficialTime << " / " << time_toc << " = " << numberOfGmresCalls << endl;
            }
        }
      }

      if (ierr > 1) {
          HPGMP_fout << "NaN Error in call to GMRES-IR: " << ierr << ".\n" << endl;
          if (A.geom->rank==0)
              std::cout << "NaN Error in call to GMRES-IR!\n" << endl;
          throw std::runtime_error("Bench GMRES-IR nan'd out!");
      }
      if (A.geom->rank==0)
      {
          std::cout << "Call [" << i << " / " << numberOfGmresCalls << "] Number of GMRES-IR Iterations ["
                  << niters <<"] Scaled Residual [" << normr/normr0 << "]\n";
        if(verbose) {
            HPGMP_fout << "Call [" << i << " / " << numberOfGmresCalls << "] Number of GMRES-IR Iterations ["
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
    test_data.optTotalFlops = test_data.flops[0];
    test_data.optTotalTime = time_solve_total;
    test_data.numOfCalls = numberOfGmresCalls;

    test_data.optNumOfMGCalls = test_data.numOfMGCalls;
    test_data.optNumOfSPCalls = test_data.numOfSPCalls;
    //test_data.opt_flops = (double*)malloc(num_flops * sizeof(double));
    //test_data.opt_times = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<n_fl_ops; i++) test_data.opt_flops[i] = test_data.flops[i];
    for (int i=0; i<n_timed_ops; i++) test_data.opt_times[i] = test_data.times[i];

    //test_data.opt_times_comp = (double*)malloc(num_times * sizeof(double));
    //test_data.opt_times_comm = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<n_timed_ops; i++) test_data.opt_times_comp[i] = test_data.times_comp[i];
    for (int i=0; i<n_timed_ops; i++) test_data.opt_times_comm[i] = test_data.times_comm[i];
  }

  // =====================================================================
  // (Optional)
  // Run reference GMRES implementation for a fixed number of iterations
  // and record the obtained Gflop/s
  if (runReference) {
    const int n_ref_calls = 10;
    //warmup
    x.fill_zero();
    GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);

    //benchmark runs
    time_solve_total = 0.0;
    for (int i=0; i<n_fl_ops; i++) test_data.flops[i] = 0.0;
    for (int i=0; i<n_timed_ops; i++) test_data.times[i] = 0.0;
    for (int i=0; i<n_timed_ops; i++) test_data.times_comp[i] = 0.0;
    for (int i=0; i<n_timed_ops; i++) test_data.times_comm[i] = 0.0;
    for (int i=0; i< n_ref_calls; ++i) {
      x.fill_zero();

      const double time_tic = mytimer();
      const int ierr = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
      const double time_toc = (mytimer() - time_tic);
      time_solve_total += time_toc;

      if (ierr == 2) {
          HPGMP_fout << "NaN Error in call to GMRES: " << ierr << ".\n" << endl;
          if (A.geom->rank==0)
              std::cout << "NaN Error in call to GMRES!" << std::endl;
      }
      if (A.geom->rank==0) {
        HPGMP_fout << "Call [" << i << " / " << n_ref_calls << "]";
        HPGMP_fout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPGMP_fout << " Time        " << time_toc << endl;
        HPGMP_fout << " Time/itr    " << time_toc / niters << endl;
        std::cout << " BenchGMRES reference Call [" << i << " / " << n_ref_calls << "]";
        std::cout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        std::cout << "     Time        " << time_toc << endl;
        std::cout << "     Time/itr    " << time_toc / niters << endl;
      }
    }
    if (verbose && A.geom->rank==0)
    {
      double flops = test_data.flops[0];
      HPGMP_fout << "  Accumulated Time " << time_solve_total << " seconds." << endl;
      HPGMP_fout << "  Final Gflop/s    " << flops/1000000000.0 << "/" << time_solve_total << " = " << (flops/1000000000.0)/time_solve_total 
                << " (n = " << A.totalNumberOfRows << ")" << endl;
    }
    test_data.refTotalFlops = test_data.flops[0];
    test_data.refTotalTime  = time_solve_total;

    test_data.refNumOfMGCalls = test_data.numOfMGCalls;
    test_data.refNumOfSPCalls = test_data.numOfSPCalls;
    //test_data.ref_flops = (double*)malloc(num_flops * sizeof(double));
    //test_data.ref_times = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<n_fl_ops; i++) test_data.ref_flops[i] = test_data.flops[i];
    for (int i=0; i<n_timed_ops; i++) test_data.ref_times[i] = test_data.times[i];

    //test_data.ref_times_comp = (double*)malloc(num_times * sizeof(double));
    //test_data.ref_times_comm = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<n_timed_ops; i++) test_data.ref_times_comp[i] = test_data.times_comp[i];
    for (int i=0; i<n_timed_ops; i++) test_data.ref_times_comm[i] = test_data.times_comm[i];
  } else {
    test_data.refTotalFlops = 0.0;
    test_data.refTotalTime  = 0.0;
  }
    
  // Report results 
  ReportResults(A, numberOfMgLevels, test_data, validation_failure);

  // cleanup
  DeleteMatrix(A);  
  DeleteMatrix(A_lo);
  DeleteGeometry(*geom);
  delete geom;

  if (verbose && A.geom->rank==0) {
    const auto total_benchmark_time = (mytimer() - benchmark_begin_time);
    HPGMP_fout << " Total benchmark time : " << total_benchmark_time << " seconds." << endl;
    std::cout << " Total benchmark time : " << total_benchmark_time << " seconds." << endl;
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform version
template
int BenchGMRES< TestGMRESData<double>, double, double, double >
  (int, char**, comm_type, DeviceCtx*, int, bool, bool, bool, TestGMRESData<double>&);

template
int BenchGMRES< TestGMRESData<float>, float, float, float >
  (int, char**, comm_type, DeviceCtx*, int, bool, bool, bool, TestGMRESData<float>&);


// mixed version
template
int BenchGMRES< TestGMRESData<double>, double, float, float >
  (int, char**, comm_type, DeviceCtx*, int, bool, bool, bool, TestGMRESData<double>&);


template<class TestGMRESDataType, class SparseMatrixType, class VectorType>
void test_mg_spmv_ref(MPI_Comm comm, DeviceCtx *const dctx, const Geometry *const geom, const SparseMatrixType& A, TestGMRESDataType& test_data)
{
    const local_int_t nrow = A.localNumberOfRows;
    const local_int_t ncol = A.localNumberOfColumns;

    VectorType x_overlap(ncol, A.comm, dctx),  // overlapped copy of x vector
               b_computed(nrow, A.comm, dctx);

    // load vector with random values
    x_overlap.fill_random();

    int ierr = 0;
    const int numberOfCalls = 10;
    const double t_begin = mytimer();
    for (int i=0; i< numberOfCalls; ++i) {
      ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
      if (ierr) HPGMP_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
      ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
      if (ierr) HPGMP_fout << "Error in call to MG: " << ierr << ".\n" << endl;
    }
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if(geom->rank == 0) {
        std::cout << "test_mg_spmv: Completed SpMV and MG refs." << std::endl;
    }
#endif
    test_data.SpmvMgTime = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
}
