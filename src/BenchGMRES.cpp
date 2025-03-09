
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
using std::endl;

#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"

#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"

#include "BenchGMRES.hpp"
#include "mytimer.hpp"

/*!
  Benchmark the optimized GMRES implementation

  @param[in]      argc      the "argc" parameter passed to the main() function
  @param[in]      argv      the "argv" parameter passed to the main() function
  @param[in]      comm      the communicator use to run benchmark

  @param[inout]   test_data the data structure with the results of the test including pass/fail information

  @return Returns zero on success and a non-zero value otherwise.

  @see GMRES()
 */


template<class TestGMRESSData_type, class scalar_type, class scalar_type2, class project_type>
int BenchGMRES(int argc, char **argv, comm_type comm, int numberOfMgLevels, bool verbose, bool runReference, TestGMRESSData_type & test_data) {

  typedef Vector<scalar_type> Vector_type;
  typedef SparseMatrix<scalar_type> SparseMatrix_type;
  typedef GMRESData<scalar_type> GMRESData_type;

  typedef Vector<scalar_type2> Vector_type2;
  typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
  typedef GMRESData<scalar_type2, project_type> GMRESData_type2;

  double total_benchmark_time = mytimer();

  //////////////////////////////////////////////////////////
  // Setup problem
  Geometry * geom = new Geometry;

  SparseMatrix_type A;
  GMRESData_type data;

  SparseMatrix_type2 A_lo;
  GMRESData_type2 data_lo;

  Vector_type b, x;
  SetupProblem("bench_",argc, argv, comm, numberOfMgLevels, verbose, geom, A, data, A_lo, data_lo, b, x, test_data);
#ifdef HPGMP_DEBUG
  MPI_Barrier(comm);
  if(geom->rank == 0) {
      std::cout << "BenchGMRES: Set up problem." << std::endl;
  }
#endif


  // =====================================================================
  // Record execution time of reference SpMV and MG kernels for reporting times
  {
    local_int_t nrow = A.localNumberOfRows;
    local_int_t ncol = A.localNumberOfColumns;

    Vector_type x_overlap, b_computed;
    InitializeVector(x_overlap, ncol, A.comm);  // Overlapped copy of x vector
    InitializeVector(b_computed, nrow, A.comm); // Computed RHS vector

    // load vector with random values
    FillRandomVector(x_overlap);
#ifdef HPGMP_DEBUG
    MPI_Barrier(comm);
    if(geom->rank == 0) {
        std::cout << "BenchGMRES: Initialized x,b vectors and filled random x." << std::endl;
    }
#endif

    int ierr = 0;
    int numberOfCalls = 10;
    double t_begin = mytimer();
    for (int i=0; i< numberOfCalls; ++i) {
      ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
#ifdef HPGMP_DEBUG
      MPI_Barrier(comm);
      if(geom->rank == 0) {
          std::cout << "BenchGMRES: Completed SPMV ref once." << std::endl;
      }
#endif
      if (ierr) HPGMP_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
      ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
#ifdef HPGMP_DEBUG
      MPI_Barrier(comm);
      if(geom->rank == 0) {
          std::cout << "BenchGMRES: Completed MG ref once." << std::endl;
      }
#endif
      if (ierr) HPGMP_fout << "Error in call to MG: " << ierr << ".\n" << endl;
    }
    test_data.SpmvMgTime = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.

    DeleteVector(x_overlap);
    DeleteVector(b_computed);
  }

  // =====================================================================
  // Benchmark parameters
  int numberOfGmresCalls = 10;
  int maxIters = 300;
  //double minOfficialTime = 1800; // Any official benchmark result must run at least this many seconds
  double minOfficialTime = 120; // for testing..
  test_data.minOfficialTime = minOfficialTime;

  int niters = 0;
  scalar_type normr (0.0);
  scalar_type normr0 (0.0);
  const scalar_type tolerance = 0.0;
  const int restart_length = test_data.restart_length;
  const bool precond = true;
  test_data.maxNumIters = maxIters;

  int num_flops = 4;
  int num_times = 12;
  test_data.flops = (double*)malloc(num_flops * sizeof(double));
  test_data.times = (double*)malloc(num_times * sizeof(double));
  test_data.times_comp = (double*)malloc(num_times * sizeof(double));
  test_data.times_comm = (double*)malloc(num_times * sizeof(double));

  // =====================================================================
  // Run optimized GMRES (here, we are calling GMRES_IR) for a fixed number of iterations
  // and record the obtained Gflop/s
  double time_solve_total = 0.0;
  {
    //warmup
    ZeroVector(x); // Zero out x
    GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
#ifdef HPGMP_DEBUG
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
    for (int i=0; i<num_flops; i++) test_data.flops[i] = 0.0;
    for (int i=0; i<num_times; i++) test_data.times[i] = 0.0;
    for (int i=0; i<num_times; i++) test_data.times_comp[i] = 0.0;
    for (int i=0; i<num_times; i++) test_data.times_comm[i] = 0.0;
    for (int i=0; i< numberOfGmresCalls; ++i) {
      ZeroVector(x); // Zero out x

      double time_tic = mytimer();
      int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x,
                          restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
#ifdef HPGMP_DEBUG
      MPI_Barrier(comm);
      if(geom->rank == 0) {
          std::cout << "BenchGMRES: Completed one benchmark GMRES-IR run." << std::endl;
      }
#endif
      double time_toc = (mytimer() - time_tic);
      time_solve_total += time_toc;
      if (i == 0) {
        if (test_data.runningTime >= 0.0) {
          int numberOfGmresCalls_min = ceil(test_data.runningTime / time_toc);
          if (numberOfGmresCalls_min > numberOfGmresCalls) {
            if (verbose && A.geom->rank==0) {
              HPGMP_fout << " numberOfGmresCalls = runningTime / time_toc = "
                        << test_data.runningTime << " / " << time_toc << " = " << numberOfGmresCalls << endl;
            }
          }
        } else {
          int numberOfGmresCalls_min = ceil(test_data.minOfficialTime / time_toc);
          if (numberOfGmresCalls_min > numberOfGmresCalls) {
            numberOfGmresCalls = numberOfGmresCalls_min;
            HPGMP_fout << " numberOfGmresCalls = minOfficialTime / time_toc = "
                      << test_data.minOfficialTime << " / " << time_toc << " = " << numberOfGmresCalls << endl;
          }
        }
      }

      if (ierr) HPGMP_fout << "Error in call to GMRES-IR: " << ierr << ".\n" << endl;
      if (verbose && A.geom->rank==0)
      {
        HPGMP_fout << "Call [" << i << " / " << numberOfGmresCalls << "] Number of GMRES-IR Iterations ["
                  << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPGMP_fout << " Time        " << time_toc << endl;
        HPGMP_fout << " Time/itr    " << time_toc / niters << endl;
      }
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
    test_data.opt_flops = (double*)malloc(num_flops * sizeof(double));
    test_data.opt_times = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<num_flops; i++) test_data.opt_flops[i] = test_data.flops[i];
    for (int i=0; i<num_times; i++) test_data.opt_times[i] = test_data.times[i];

    test_data.opt_times_comp = (double*)malloc(num_times * sizeof(double));
    test_data.opt_times_comm = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<num_times; i++) test_data.opt_times_comp[i] = test_data.times_comp[i];
    for (int i=0; i<num_times; i++) test_data.opt_times_comm[i] = test_data.times_comm[i];
  }

  // =====================================================================
  // (Optional)
  // Run reference GMRES implementation for a fixed number of iterations
  // and record the obtained Gflop/s
  if (runReference) {
    //warmup
    ZeroVector(x); // Zero out x
    GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);

    //benchmark runs
    time_solve_total = 0.0;
    for (int i=0; i<num_flops; i++) test_data.flops[i] = 0.0;
    for (int i=0; i<num_times; i++) test_data.times[i] = 0.0;
    for (int i=0; i<num_times; i++) test_data.times_comp[i] = 0.0;
    for (int i=0; i<num_times; i++) test_data.times_comm[i] = 0.0;
    for (int i=0; i< numberOfGmresCalls; ++i) {
      ZeroVector(x); // Zero out x

      double time_tic = mytimer();
      int ierr = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
      double time_toc = (mytimer() - time_tic);
      time_solve_total += time_toc;

      if (ierr) HPGMP_fout << "Error in call to GMRES: " << ierr << ".\n" << endl;
      if (verbose && A.geom->rank==0) {
        HPGMP_fout << "Call [" << i << " / " << numberOfGmresCalls << "]";
        HPGMP_fout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPGMP_fout << " Time        " << time_toc << endl;
        HPGMP_fout << " Time/itr    " << time_toc / niters << endl;
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
    test_data.ref_flops = (double*)malloc(num_flops * sizeof(double));
    test_data.ref_times = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<num_flops; i++) test_data.ref_flops[i] = test_data.flops[i];
    for (int i=0; i<num_times; i++) test_data.ref_times[i] = test_data.times[i];

    test_data.ref_times_comp = (double*)malloc(num_times * sizeof(double));
    test_data.ref_times_comm = (double*)malloc(num_times * sizeof(double));
    for (int i=0; i<num_times; i++) test_data.ref_times_comp[i] = test_data.times_comp[i];
    for (int i=0; i<num_times; i++) test_data.ref_times_comm[i] = test_data.times_comm[i];
  } else {
    test_data.refTotalFlops = 0.0;
    test_data.refTotalTime  = 0.0;
  }

  // cleanup
  DeleteMatrix(A);  
  DeleteMatrix(A_lo);
  DeleteGeometry(*geom);
  delete geom;

  DeleteGMRESData(data);
  DeleteGMRESData(data_lo);
  DeleteVector(x);
  DeleteVector(b);

  if (verbose && A.geom->rank==0) {
    total_benchmark_time = (mytimer() - total_benchmark_time);
    HPGMP_fout << " Total benchmark time : " << total_benchmark_time << " seconds." << endl;
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform version
template
int BenchGMRES< TestGMRESData<double>, double, double, double >
  (int, char**, comm_type, int, bool, bool, TestGMRESData<double>&);

template
int BenchGMRES< TestGMRESData<float>, float, float, float >
  (int, char**, comm_type, int, bool, bool, TestGMRESData<float>&);


// mixed version
template
int BenchGMRES< TestGMRESData<double>, double, float, float >
  (int, char**, comm_type, int, bool, bool, TestGMRESData<double>&);

