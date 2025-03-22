
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

// Main routine of a program that calls the HPGMP GMRES and GMRES-IR 
// solvers to solve the problem, and then prints results.

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>

#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "Geometry.hpp"
//#include "SparseMatrix.hpp"
#include "device_ctx.hpp"
#include "Vector.hpp"

#include "GMRESData.hpp"
#include "ValidGMRES.hpp"
#include "BenchGMRES.hpp"
#include "mytimer.hpp"

using scalar_type =  double;
using scalar_type2 = float;
using project_type = float;

typedef TestGMRESData<scalar_type> TestGMRESData_type;
typedef Vector<scalar_type> Vector_type;
typedef SparseMatrix<scalar_type> SparseMatrix_type;
typedef GMRESData<scalar_type> GMRESData_type;

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

  int myRank = 0;
#ifndef HPGMP_NO_MPI
  int numRanks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
#endif

  auto ctx = std::make_unique<DeviceCtx>(myRank);

  //////////////////////////
  // Create Communicators //
  //////////////////////////
  int sizeValidComm = 8;
#ifndef HPGMP_NO_MPI
  int color = 0;
  if (sizeValidComm > numRanks) {
    sizeValidComm = numRanks;
  }
  if (myRank < sizeValidComm) {
    color = 1;
  }
  MPI_Comm validation_comm = MPI_COMM_WORLD;
  MPI_Comm benchmark_comm = MPI_COMM_WORLD;
  MPI_Comm_split(MPI_COMM_WORLD, color, myRank, &validation_comm);
# ifdef HPGMP_VERBOSE
  if (myRank == 0) {
      std::cout << "main: created split validation comm." << std::endl;
  }
# endif
#else
  comm_type validation_comm = 0;
  comm_type benchmark_comm = 0;
#endif

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  const int numberOfMgLevels = 4; // Number of levels including first
 
  std::string working_precision = "double", inner_precision = "double", project_precision = "double";
  if(std::is_same<scalar_type,float>::value) {
      working_precision = "float";
  }
  if(std::is_same<scalar_type2,float>::value) {
      inner_precision = "float";
  }
  if(std::is_same<project_type,float>::value) {
      project_precision = "float";
  }
  if(myRank == 0) {
      std::cout << "Running HPG-MxP benchmark with working precision " << working_precision
                << ",\n  inner precision " << inner_precision
                << " and projection precision " << project_precision << "." << std::endl;
  }


#ifdef HPGMP_DEBUG
  const bool verbose = true;
#else
  const bool verbose = false;
#endif

  // Use this array for collecting timing information
  TestGMRESData_type test_data;
  //test_data.times = NULL;
  //test_data.flops = NULL;
  test_data.validation_nprocs = sizeValidComm;


  //////////////////////
  // Validation phase //
  //////////////////////
  int global_failure = 0;
  const int restart_length = 40;
  const scalar_type tolerance = 1e-9;

  test_data.tolerance = tolerance;
  test_data.restart_length = restart_length;
  if (myRank < sizeValidComm) {
    global_failure = ValidGMRES<TestGMRESData_type, scalar_type, scalar_type2, project_type>
                         (argc, argv, validation_comm, ctx.get(), numberOfMgLevels, verbose, test_data);
  }


  /////////////////////
  // Benchmark phase //
  /////////////////////
  {
    bool runReference = true;
    BenchGMRES<TestGMRESData_type, scalar_type, scalar_type2, project_type>
        (argc, argv, benchmark_comm, ctx.get(), numberOfMgLevels, verbose, runReference, global_failure, test_data);
#ifndef HPGMP_NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

 
#if 0 
  ////////////////////
  // Report Results //
  ////////////////////
  {
    // setup problem for reporting (TODO: remove)
    Geometry * geom = new Geometry;

    SparseMatrix_type A;
    GMRESData_type data;

    SparseMatrix_type2 A2;
    GMRESData_type2 data2;

    Vector_type b, x;
    SetupProblem("report_", argc, argv, benchmark_comm, ctx.get(), numberOfMgLevels, verbose, geom, A, data, A2, data2, b, x, test_data);


    // Report results to YAML file
    ReportResults(A, numberOfMgLevels, test_data, global_failure);

    // Clean up
    DeleteMatrix(A);
    DeleteMatrix(A2);
    DeleteGeometry(*geom);
    delete geom;
  }
#endif

  HPGMP_Finalize();
#ifndef HPGMP_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
