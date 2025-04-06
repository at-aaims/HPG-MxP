
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
 @file TestGMRES.cpp

 HPGMP routine
 */

// Changelog
// None
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"

#include "ValidGMRES.hpp"
#include "mytimer.hpp"

/*!
  Test the correctness of the optimized GMRES implementation

  @param[in]      argc      the "argc" parameter passed to the main() function
  @param[in]      argv      the "argv" parameter passed to the main() function
  @param[in]      comm      the communicator used to run validation
  @param[inout]   test_data the data structure with the results of the test including pass/fail information

  @return Returns zero on success and a non-zero value otherwise.

  @see GMRES()
 */


template<class TestGMRESSData_type, class scalar_type, class scalar_type2, class project_type>
int ValidGMRES(const int argc, char **argv, comm_type comm, DeviceCtx *const dctx, const int numberOfMgLevels, const bool verbose, TestGMRESSData_type & test_data) {

  typedef Vector<scalar_type> Vector_type;
  typedef SparseMatrix<scalar_type> SparseMatrix_type;
  typedef GMRESData<scalar_type> GMRESData_type;

  typedef Vector<scalar_type2> Vector_type2;
  typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
  typedef GMRESData<scalar_type2,  project_type> GMRESData_type2;

  double total_validation_time = mytimer();

  //////////////////////////////////////////////////////////
  // Setup problem
  Geometry * geom = new Geometry;

  SparseMatrix_type A;
  GMRESData_type data;

  SparseMatrix_type2 A_lo;
  GMRESData_type2 data_lo;

  Vector_type b, x;
  SetupProblem("valid_", argc, argv, comm, dctx, numberOfMgLevels, verbose, geom, A, data, A_lo, data_lo, b, x, test_data);

  //////////////////////////////////////////////////////////
  // Solver Parameters
  const int MaxIters = 10000;
  const int restart_length = test_data.restart_length;
  const scalar_type tolerance = test_data.tolerance;
  if (A.geom->rank == 0) {
      std::cout << " Validate GMRES( tol = " << tolerance << " and restart = " << restart_length << " ) <<" << std::endl;
    HPGMP_fout << std::endl << " >> In Validate GMRES( tol = " << tolerance << " and restart = " << restart_length << " ) <<" << std::endl;
  }


  //////////////////////////////////////////////////////////
  // Run reference GMRES to a fixed tolerance
  int fail = 0;
  int refNumIters = 0;
  double refSolveTime = 0.0;
  scalar_type refResNorm = 0.0;
  scalar_type refResNorm0 = 0.0;
  {
    x.fill_zero();

    const double time_tic = mytimer();
    const int ierr = GMRES(A, data, b, x, restart_length, MaxIters, tolerance,
                           refNumIters, refResNorm, refResNorm0, true, verbose, test_data);
    refSolveTime = (mytimer() - time_tic);
    fail = ierr;
    if (ierr && A.geom->rank == 0) {
        std::cout << " ValidGMRES: Error in call to ref GMRES: " << ierr << ".\n" << std::endl;
        if (A.geom->rank == 0)
            throw std::runtime_error("Ref GMRES did not converge to specified tolerence!");
    }

    test_data.refNumIters = refNumIters;
    test_data.refResNorm0 = refResNorm0;
    test_data.refResNorm  = refResNorm;
  }
  if (A.geom->rank==0) {
    HPGMP_fout << "  Reference Iteration time  " << refSolveTime << " seconds." << std::endl;
    HPGMP_fout << "  Reference Iteration count " << refNumIters << std::endl;
    std::cout << " ValidGMRES: Reference initial norm = " << refResNorm0 << std::endl;
    std::cout << " ValidGMRES: Reference final norm = " << refResNorm << std::endl;
    std::cout << " ValidGMRES: Reference Iteration time  " << refSolveTime << " seconds." << std::endl;
    std::cout << " ValidGMRES: Reference Iteration count " << refNumIters << std::endl;
  }
  if (A.geom->rank == 0 && refResNorm/refResNorm0 > tolerance) {
    HPGMP_fout << " ref GMRES did not converege: normr = "
        << refResNorm << " / " << refResNorm0 << " = " << refResNorm/refResNorm0
        << "(tol = " << tolerance << ")" << std::endl;
    std::cout << " ValidGMRES: ref GMRES did not converege: normr = "
        << refResNorm << " / " << refResNorm0 << " = " << refResNorm/refResNorm0
        << "(tol = " << tolerance << ")" << std::endl;
  }


  //////////////////////////////////////////////////////////
  // Run "optimized" GMRES (aka GMRES-IR) to a fixed tolerance
  int optNumIters = 0;
  double optSolveTime = 0.0;
  scalar_type optResNorm = 0.0;
  scalar_type optResNorm0 = 0.0;
  {
    x.fill_zero();

    double time_tic = mytimer();
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if(A.geom->rank == 0) {
        std::cout << "ValidGMRES: Starting optimized GMRES-IR" << std::endl;
    }
#endif
    const int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, MaxIters, tolerance,
                              optNumIters, optResNorm, optResNorm0, true, verbose, test_data);
    optSolveTime = (mytimer() - time_tic);
    fail = ierr;
    if (ierr) {
        if(A.geom->rank == 0)
            std::cout << " ValidGMRES: Error in call to opt GMRES-IR: " << ierr << ".\n" << std::endl;
        throw std::runtime_error("Opt GMRES did not converge to specified tolerence!");
    }

    test_data.optNumIters = optNumIters;
    test_data.optResNorm0 = optResNorm0;
    test_data.optResNorm  = optResNorm;
  }
  if (A.geom->rank==0) {
      std::cout << " ValidGMRES: Optimized initial residual norm  " << optResNorm0 << "\n";
      std::cout << " ValidGMRES: Optimized final residual norm = " << optResNorm << std::endl
                << "             Optimized iteration time  " << optSolveTime << " seconds,\n" 
                << "             optimized iteration count = " << optNumIters << std::endl;
  }
  if (verbose && A.geom->rank==0) {
      HPGMP_fout << "  Optimized Iteration time  " << optSolveTime << " seconds." << std::endl;
      HPGMP_fout << "  Optimized Iteration count " << optNumIters << std::endl;
  }
  if (optResNorm/optResNorm0 > tolerance) {
    fail = 3;
    if (A.geom->rank == 0) {
      HPGMP_fout << " opt GMRES did not converege: normr = "
          << optResNorm << " / " << optResNorm0 << " = " << optResNorm/optResNorm0
          << "(tol = " << tolerance << ")" << std::endl;
      std::cout << " ValidGMRES: opt GMRES did not converege: normr = "
          << optResNorm << " / " << optResNorm0 << " = " << optResNorm/optResNorm0
          << "(tol = " << tolerance << ")" << std::endl;
    }
  }


  // cleanup
  DeleteMatrix(A);
  DeleteMatrix(A_lo);
#ifdef HPGMP_VERBOSE
  MPI_Barrier(comm);
  if(A.geom->rank == 0) {
      std::cout << "ValidGMRES: Completed optimized GMRES-IR." << std::endl;
  }
#endif
  if (verbose && A.geom->rank==0) {
    total_validation_time = (mytimer() - total_validation_time);
    HPGMP_fout << " Total validation time : " << total_validation_time << " seconds." << std::endl;
  }

  delete geom;
  return fail;
}



/* --------------- *
 * specializations *
 * --------------- */

// uniform version
template
int ValidGMRES<TestGMRESData<double>,  double, double, double > (int, char**, comm_type, DeviceCtx*, int, bool, TestGMRESData<double>&);

template
int ValidGMRES< TestGMRESData<float>, float, float, float > (int, char**, comm_type, DeviceCtx*, int, bool, TestGMRESData<float>&);

// mixed version
template
int ValidGMRES< TestGMRESData<double>, double, float, float > (int, char**, comm_type, DeviceCtx*, int, bool, TestGMRESData<double>&);

