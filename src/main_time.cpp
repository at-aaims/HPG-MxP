
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
 @file main.cpp

 HPGMP routine
 */

// Main routine of a program that calls the HPGMP conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPGMP_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>
#include <memory>

#include "hpgmp.hpp"

#include "SetupMatrix.hpp"
#include "CheckAspectRatio.hpp"
#include "CheckProblem.hpp"
#include "OptimizeProblem.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

#include "TestGMRES.hpp"

typedef double scalar_type;
//typedef float  scalar_type;

typedef Vector<scalar_type> Vector_type;
typedef SparseMatrix<scalar_type> SparseMatrix_type;
typedef GMRESData<scalar_type> GMRESData_type;

typedef float scalar_type2;
typedef float project_type;
typedef Vector<scalar_type2> Vector_type2;
typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
typedef GMRESData<scalar_type2, project_type> GMRESData_type2;


/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPGMP_NO_MPI
  MPI_Init(&argc, &argv);
#endif
  HPGMP_Init(&argc, &argv);
#ifndef HPGMP_NO_MPI
  MPI_Comm bench_comm = MPI_COMM_WORLD;
#else
  comm_type bench_comm = 0;
#endif

  HPGMP_Params params;
  HPGMP_Init_Params(&argc, &argv, params, bench_comm);
  const int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID
  if (rank == 0) HPGMP_fout << endl;
#ifndef HPGMP_NO_MPI
  if (rank == 0) HPGMP_fout << "With MPI " << endl;
#endif
#ifdef HPGMP_WITH_CUDA
  if (rank == 0) HPGMP_fout << "With Cuda " << endl;
#endif
  if (rank == 0) HPGMP_fout << endl;
  
  auto dctx = std::make_unique<DeviceCtx>(rank);

#ifdef HPGMP_DETAILED_DEBUG
  if (size < 100 && rank==0) HPGMP_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

  if (rank==0) {
    char c;
    std::cout << "Press key to continue"<< std::endl;
    std::cin.get(c);
  }
#ifndef HPGMP_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0);
  if (ierr)
    return ierr;

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

#ifdef HPGMP_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);
  if (ierr)
    return ierr;

  // Use this array for collecting timing information
  std::vector< double > times(10,0.0);

  double setup_time = mytimer();

  // Setup the problem
  SparseMatrix_type A;
  GMRESData_type data;

  bool init_vect = true;
  Vector_type b, x, xexact;

  int numberOfMgLevels = 4; // Number of levels including first
  SetupMatrix(dctx.get(), numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect, bench_comm);

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;

  if (A.geom->rank==0) {
    HPGMP_fout << " Setup    Time     " << setup_time << " seconds." << endl;
    HPGMP_fout << " Optimize Time     " << t7 << " seconds." << endl;
  }

  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  const local_int_t nrow = A.localNumberOfRows;
  const local_int_t ncol = A.localNumberOfColumns;

  Vector_type b_computed(nrow, bench_comm, dctx.get()); // Computed RHS vector


  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////
  bool test_diagonal_exaggeration = false;
  bool test_noprecond = true;
  TestGMRESData test_data;

#ifdef HPGMP_DEBUG
  t1 = mytimer();
  if (rank==0) HPGMP_fout << endl << "Running Uniform-precision Test" << endl;
#endif
  TestGMRES(A, data, b, x, test_diagonal_exaggeration, test_noprecond, test_data);
#ifdef HPGMP_DEBUG
  if (rank==0) HPGMP_fout << "Total validation (uniform-precision TestGMRES) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  setup_time = mytimer();
  init_vect = false;
  SparseMatrix_type2 A2;
  GMRESData_type2 data2;
  SetupMatrix(dctx.get(), numberOfMgLevels, A2, geom, data2, &b, &x, &xexact, init_vect, bench_comm);
  setup_time = mytimer() - setup_time; // Capture total time of setup

  t7 = mytimer();
  OptimizeProblem(A2, data, b, x, xexact);
  t7 = mytimer() - t7;

  if (A.geom->rank==0) {
    HPGMP_fout << " Setup    Time     " << setup_time << " seconds." << endl;
    HPGMP_fout << " Optimize Time     " << t7 << " seconds." << endl;
  }


#ifdef HPGMP_DEBUG
  t1 = mytimer();
#endif
  TestGMRES(A, A2, data, data2, b, x, test_diagonal_exaggeration, test_noprecond, test_data);
#ifdef HPGMP_DEBUG
  if (rank==0) HPGMP_fout << "Total validation (mixed-precision TestGMRES) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  HPGMP_Finalize();
#ifndef HPGMP_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
