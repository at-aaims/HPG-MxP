
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// ***************************************************
//@HEADER

#include <iostream>
#include <cstdlib>

#include <vector>
#include <memory>

#include "hpgmp.hpp"

#include "SetupMatrix.hpp"
#include "CheckAspectRatio.hpp"
#include "CheckProblem.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"
#include "ell_matrix.hpp"
#include "simulate_halos.hpp"

typedef double scalar_type;
//typedef float  scalar_type;

typedef Vector<scalar_type> Vector_type;
typedef SparseMatrix<scalar_type> SparseMatrix_type;
typedef GMRESData<scalar_type, scalar_type, scalar_type> GMRESData_type;

typedef float scalar_type2;
typedef float project_type;
typedef Vector<scalar_type2> Vector_type2;
#ifdef HPGMP_WITH_GINKGO
typedef SparseMatrix<scalar_type, scalar_type2> SparseMatrix_type2;
typedef GMRESData<scalar_type, scalar_type2, project_type> GMRESData_type2;
#else
typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
typedef GMRESData<scalar_type2, scalar_type2, project_type> GMRESData_type2;
#endif

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char* argv[])
{
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
    params.numThreads = 1;
    const int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

    auto dctx = std::make_unique<DeviceCtx>(rank);

    const local_int_t nx = 16;
    const local_int_t ny = 16;
    const local_int_t nz = 16;
    int ierr             = 0; // Used to check return codes on function calls

    ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank == 0);
    if (ierr) {
        return ierr;
    }

    /////////////////////////
    // Problem setup Phase //
    /////////////////////////

#ifdef HPGMP_DEBUG
    double t1 = mytimer();
#endif

    // Construct the geometry and linear system
    Geometry* geom = new Geometry;
    GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

    ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank == 0);
    if (ierr) {
        return ierr;
    }

    // Use this array for collecting timing information
    std::vector<double> times(10, 0.0);

    double setup_time = mytimer();

    // Setup the problem
    SparseMatrix_type A;
    GMRESData_type data;

    bool init_vect = true;
    Vector_type b, x, xexact;

    const int numberOfMgLevels = 1; // Number of levels including first
    SetupMatrix(dctx.get(), numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect, bench_comm);

    setup_time = mytimer() - setup_time; // Capture total time of setup
    times[9]   = setup_time; // Save it for reporting

    // Simulate halos
    const local_int_t nhalos = simulate_halos<SparseMatrix_type>(A);

    // Call user-tunable set up function.
    double t7 = mytimer();
    OptimizeProblem(A, data, b, x, xexact);
    t7       = mytimer() - t7;
    times[7] = t7;

    if (A.geom->rank == 0) {
        std::cout << " Setup    Time     " << setup_time << " seconds." << std::endl;
        std::cout << " Optimize Time     " << t7 << " seconds." << std::endl;
    }

    // free
    DeleteMatrix(A);
    delete geom;
    HPGMP_Finalize();
#ifndef HPGMP_NO_MPI
    MPI_Finalize();
#endif
    return 0;
}
