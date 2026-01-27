
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
 @file GenerateProblem.cpp

 HPGMP routine
 */

#include <iostream>

#include "hpgmp.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include "SetupMatrix.hpp"
#include "SetupProblem.hpp"
#include "CheckAspectRatio.hpp"
#include "OptimizeProblem.hpp"

#include "mytimer.hpp"
using std::endl;

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type>
void SetupProblem(const char* title, int argc, char** argv, comm_type comm, DeviceCtx* const dctx,
                  int numberOfMgLevels, bool verbose, Geometry* geom, SparseMatrix_type& A,
                  GMRESData_type& data, SparseMatrix_type2& A2, GMRESData_type2& data2,
                  Vector_type& b, Vector_type& x, TestGMRESData& test_data)
{
    HPGMP_Params params;
    HPGMP_Init_Params(title, &argc, &argv, params, comm);
    const int size        = params.comm_size; // Number of MPI processes
    const int rank        = params.comm_rank; // My process ID
    test_data.runningTime = params.runningTime;

    local_int_t nx = (local_int_t)params.nx;
    local_int_t ny = (local_int_t)params.ny;
    local_int_t nz = (local_int_t)params.nz;

    //////////////////////////////////////////////////////////
    // Construct the geometry and linear system
    GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu,
                     nx, ny, nz, params.npx, params.npy, params.npz, geom);
#ifdef HPGMP_DEBUG
    MPI_Barrier(comm);
    if (rank == 0) {
        std::cout << "   SetupProblem: generated geometry." << std::endl;
    }
#endif
    int ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank == 0);


    //////////////////////////////////////////////////////////
    // Setup the problem
    bool init_vect = true;
    Vector_type xexact;
    double setup_time = mytimer();
    SetupMatrix(dctx, numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect, comm);
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if (rank == 0) {
        std::cout << "   SetupProblem: set up DP matrix." << std::endl;
    }
#endif

    // Setup single-precision A
    init_vect = false;
    SetupMatrix(dctx, numberOfMgLevels, A2, geom, data2, &b, &x, &xexact, init_vect, comm);

    setup_time = mytimer() - setup_time; // Capture total time of setup
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if (rank == 0) {
        std::cout << "   SetupProblem: set up LP matrix." << std::endl;
    }
#endif
    //times[9] = setup_time; // Save it for reporting
    test_data.SetupTime = setup_time;

    //////////////////////////////////////////////////////////
    // Call user-tunable set up function for A
    double opt_time = mytimer();
    OptimizeProblem(A, data, b, x, xexact);
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if (rank == 0) {
        std::cout << "   SetupProblem: Optimized DP problem." << std::endl;
    }
#endif

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
    A.delete_host_data();
#endif

    // Call user-tunable set up function for A2
    OptimizeProblem(A2, data, b, x, xexact);
    opt_time = mytimer() - opt_time; // Capture total time of setup
#ifdef HPGMP_VERBOSE
    MPI_Barrier(comm);
    if (rank == 0) {
        std::cout << "   SetupProblem: Optimized LP problem." << std::endl;
    }
#endif

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
    A2.delete_host_data();
#endif

    //times[7] = opt_time;
    test_data.OptimizeTime = opt_time;

    if (verbose && A.geom->rank == 0) {
        HPGMP_fout << " Setup    Time     " << setup_time << " seconds." << endl;
        HPGMP_fout << " Optimize Time     " << opt_time << " seconds." << endl;
    }
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template void SetupProblem< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>>(
    const char*, int, char**, comm_type, DeviceCtx*, int, bool, Geometry*, SparseMatrix<double>&,
    GMRESData<double>&, SparseMatrix<double>&, GMRESData<double>&,
    Vector<double>&, Vector<double>&, TestGMRESData&);

template void SetupProblem< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>>(
    const char*, int, char**, comm_type, DeviceCtx*, int, bool, Geometry*, SparseMatrix<float>&,
    GMRESData<float>&, SparseMatrix<float>&, GMRESData<float>&,
    Vector<float>&, Vector<float>&, TestGMRESData&);

// mixed
template void SetupProblem< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>>(
    const char*, int, char**, comm_type, DeviceCtx*, int, bool, Geometry*, SparseMatrix<double>&,
    GMRESData<double>&, SparseMatrix<float>&, GMRESData<float>&,
    Vector<double>&, Vector<double>&, TestGMRESData&);
