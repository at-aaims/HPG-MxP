
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

using scalar_type  = double;
using scalar_type2 = float;
using project_type = float;

typedef Vector<scalar_type> Vector_type;
typedef SparseMatrix<scalar_type> SparseMatrix_type;
typedef GMRESData<scalar_type> GMRESData_type;

typedef Vector<scalar_type2> Vector_type2;
typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
typedef GMRESData<scalar_type2, project_type> GMRESData_type2;

int get_valid_comm_size(const HPGMP_gen_opts& gopts, const int global_size)
{
    if (gopts.validation_type == validation_t::standard && global_size >= 8) {
        return 8;
    } else {
        return global_size;
    }
}

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char* argv[])
{

#ifndef HPGMP_NO_MPI
    int provided_thread_support = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_support);
    if (provided_thread_support != MPI_THREAD_FUNNELED) {
        printf("!!Unsuppored threading mode!!\n");
        fflush(stdout);
    }
#endif
    const HPGMP_gen_opts gopts = HPGMP_Init(&argc, &argv);

    double t0{};
    TICK();

    int myRank   = 0;
    int numRanks = 1;
#ifndef HPGMP_NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
#endif

    auto ctx = std::make_unique<DeviceCtx>(myRank);

    //////////////////////////
    // Create Communicators //
    //////////////////////////
#ifndef HPGMP_NO_MPI
    comm_type benchmark_comm  = MPI_COMM_WORLD;
    comm_type validation_comm = MPI_COMM_WORLD;
    const int sizeValidComm   = get_valid_comm_size(gopts, numRanks);
    if (gopts.validation_type == validation_t::standard) {
        int color = 0;
        if (myRank < sizeValidComm) {
            color = 1;
        }
        MPI_Comm_split(MPI_COMM_WORLD, color, myRank, &validation_comm);
        if (myRank == 0) {
            std::cout << "main: Using standard validation.\n";
        }
    } else {
        if (myRank == 0) {
            std::cout << "main: Using global-scale validation.\n";
        }
    }

    if (gopts.run_type == run_t::benchmark) {
        if (myRank == 0) {
            std::cout << "Running full benchmark mode." << std::endl;
        }
    } else if (gopts.run_type == run_t::benchmark_no_ref) {
        if (myRank == 0) {
            std::cout << "Running benchmark mode without reference (DP) runs." << std::endl;
        }
    } else if (gopts.run_type == run_t::standalone_ref) {
        if (myRank == 0) {
            std::cout << "Running standalone reference (DP) mode." << std::endl;
        }
    } else {
        if (myRank == 0) {
            std::cout << "Running standalone MxP mode." << std::endl;
        }
    }

    int ierr = MPI_Comm_set_errhandler(validation_comm, MPI_ERRORS_RETURN);
    if (ierr != MPI_SUCCESS) {
        printf("! Could not set MPI error handler on validation!\n");
        fflush(stdout);
    }
    ierr = MPI_Comm_set_errhandler(benchmark_comm, MPI_ERRORS_RETURN);
    if (ierr != MPI_SUCCESS) {
        printf("! Could not set MPI error handler!\n");
        fflush(stdout);
    }
#ifdef HPGMP_VERBOSE
    if (myRank == 0) {
        std::cout << "main: created split validation comm." << std::endl;
    }
#endif
#else
    comm_type validation_comm = 0;
    comm_type benchmark_comm  = 0;
    const int sizeValidComm   = 1;
#endif // HPGMP_NO_MPI

    // Check if QuickPath option is enabled.
    // If the running time is set to zero, we minimize all paths through the program
    const int numberOfMgLevels = 4; // Number of levels including first

    std::string working_precision = "double", inner_precision = "double", project_precision = "double";
    if (std::is_same<scalar_type, float>::value) {
        working_precision = "float";
    }
    if (std::is_same<scalar_type2, float>::value) {
        inner_precision = "float";
    }
    if (std::is_same<project_type, float>::value) {
        project_precision = "float";
    }
    if (myRank == 0) {
        std::cout << "Running HPG-MxP benchmark with working precision " << working_precision
                  << ",\n  inner precision " << inner_precision
                  << " and projection precision " << project_precision << "." << std::endl;
    }

    double t_setup{};
    TOCK(t_setup);
    if (myRank == 0) {
        std::cout << "Main: setup took " << t_setup << " s." << std::endl;
    }


#ifdef HPGMP_VERBOSE
    const bool verbose = true;
#else
    const bool verbose = false;
#endif

    // Use this array for collecting timing information
    TestGMRESData test_data;
    //test_data.times = NULL;
    //test_data.flops = NULL;
    test_data.validation_nprocs = sizeValidComm;


    //////////////////////
    // Validation phase //
    //////////////////////
    int global_failure          = 0;
    const int restart_length    = 30;
    const scalar_type tolerance = 1e-9;
    test_data.restart_length    = restart_length;
    test_data.tolerance         = tolerance;
    const bool to_validate      = (gopts.run_type == run_t::benchmark || gopts.run_type == run_t::benchmark_no_ref);

    if (myRank < sizeValidComm && to_validate) {
        TICK();
        global_failure = ValidGMRES<scalar_type, scalar_type2, project_type>(
            argc, argv, gopts.validation_type, validation_comm, ctx.get(),
            numberOfMgLevels, verbose, test_data);
        double t_valid{};
        TOCK(t_valid);
        if (myRank == 0) {
            std::cout << "Main: Validation took " << t_valid << " s." << std::endl;
        }
    }

    if (!to_validate) {
        test_data.refNumIters = test_data.optNumIters = 1;
    }


    /////////////////////
    // Benchmark phase //
    /////////////////////
    {
        const bool runReference = true;
        TICK();
        BenchGMRES<scalar_type, scalar_type2, project_type>(argc, argv,
                                                            benchmark_comm, ctx.get(), numberOfMgLevels, verbose, global_failure, gopts,
                                                            test_data);
        double t_bench{};
        TOCK(t_bench);
        if (myRank == 0) {
            std::cout << "Main: Benchmarking, reference run and reporting took " << t_bench << " s."
                      << std::endl;
        }
    }

    HPGMP_Finalize();
#ifndef HPGMP_NO_MPI
    MPI_Finalize();
#endif
    return 0;
}
