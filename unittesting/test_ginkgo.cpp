
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
#include "GinkgoInterface.hpp"

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
#ifndef HPGMP_NO_MPI
    //const local_int_t nhalos = simulate_halos<SparseMatrix_type>(A);
#endif

    // Call user-tunable set up function.
    double t7 = mytimer();
    OptimizeProblem(A, data, b, x, xexact);
    t7       = mytimer() - t7;
    times[7] = t7;

#ifdef HPGMP_VERBOSE
    std::cout << "A.localNumberOfRows: " << A.localNumberOfRows << "\n";
    std::cout << "A.localNumberOfColumns: " << A.localNumberOfColumns << "\n";
    std::cout << "A.localNumberOfNonzeros: " << A.localNumberOfNonzeros << "\n";
    std::cout << "A.max_nnz_per_row: " << A.max_nnz_per_row << "\n";
    std::cout << "A.colInd: (First 10 rows)\n";
    for (local_int_t row = 0; row < 10; ++row)
    {
        auto currentRowColInd = A.mtxIndL[row];
        for (local_int_t col = 0; col < A.max_nnz_per_row; ++col)
        {
            std::cout << currentRowColInd[col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "A.matrixValues: (First 10 rows)\n";
    for (local_int_t row = 0; row < 10; ++row)
    {
        auto currentRowValues = A.matrixValues[row];
        auto currentRowColInd = A.mtxIndL[row];
        for (local_int_t col = 0; col < A.max_nnz_per_row; ++col)
        {
            std::cout << currentRowValues[col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "b.values(): (First 10 rows)\n";
    {
        auto b_values = b.values();
        for (local_int_t row = 0; row < 10; ++row)
        {
            std::cout << b_values[row] << "\n";
        }
    }
    std::cout << "xexact.values(): (First 10 rows)\n";
    {
        auto xexact_values = xexact.values();
        for (local_int_t row = 0; row < 10; ++row)
        {
            std::cout << xexact_values[row] << "\n";
        }
    }
#endif

    // Use Ginkgo to solve
    double ginkgo_time       = mytimer();
    using ginkgo_ell_type    = gko::matrix::Ell<scalar_type, local_int_t>;
    using ginkgo_coo_type    = gko::matrix::Coo<scalar_type, local_int_t>;
    using ginkgo_vec_type    = gko::matrix::Dense<scalar_type>;
    using gmres              = gko::solver::Gmres<>;
    using bj                 = gko::preconditioner::Jacobi<>;
    auto ginkgo_mat          = ginkgo_coo_type::create(ginkgo_exec, gko::dim<2>{A.localNumberOfRows, A.localNumberOfColumns}, A.localNumberOfNonzeros);
    auto rhs                 = ginkgo_vec_type::create(ginkgo_exec, gko::dim<2>{A.localNumberOfRows, 1}, 1);
    auto u                   = ginkgo_vec_type::create(ginkgo_exec, gko::dim<2>{A.localNumberOfRows, 1}, 1);
    auto ginkgo_mat_values   = ginkgo_mat->get_values();
    auto ginkgo_mat_rows     = ginkgo_mat->get_row_idxs();
    auto ginkgo_mat_cols     = ginkgo_mat->get_col_idxs();
    auto rhs_values          = rhs->get_values();
    auto b_values            = b.values();
    local_int_t i = 0;
    for (local_int_t row = 0; row < A.localNumberOfColumns; ++row)
    {
        auto currentRowValues = A.matrixValues[row];
        auto currentRowColInd = A.mtxIndL[row];
        for (local_int_t col = 0; col < A.max_nnz_per_row; ++col)
        {
            if (currentRowColInd[col] != -1)
            {
              ginkgo_mat_rows[i] = row;
              ginkgo_mat_cols[i] = currentRowColInd[col];
              ginkgo_mat_values[i] = currentRowValues[col];
              i++;
            }
        }
        rhs_values[row] = b_values[row];
    }
    auto solver_factory = gmres::build()
                              .with_criteria(gko::stop::Iteration::build()
                                                 .with_max_iters(1000)
                                                 .on(ginkgo_exec),
                                             gko::stop::ResidualNorm<scalar_type>::build()
                                                 .with_reduction_factor(1e-9)
                                                 .on(ginkgo_exec))
                              .with_preconditioner(bj::build().with_max_block_size(8u).on(ginkgo_exec))
                              .on(ginkgo_exec);
    auto solver = solver_factory->generate(gko::clone(ginkgo_exec, ginkgo_mat));
    std::shared_ptr<const gko::log::Convergence<scalar_type>> logger = gko::log::Convergence<scalar_type>::create();
    solver->add_logger(logger);
    solver->apply(rhs, u);
    ginkgo_time = mytimer() - ginkgo_time;

#ifdef HPGMP_VERBOSE
    auto u_values = u->get_values();
    std::cout << "u.values(): (First 10 rows)\n";
    for (local_int_t i = 0; i < 10; ++i)
    {
        std::cout << u_values[i] << "\n";
    }
#endif

    if (A.geom->rank == 0) {
        std::cout << " Setup    Time               " << setup_time << " seconds." << std::endl;
        std::cout << " Optimize Time               " << t7 << " seconds." << std::endl;
        std::cout << " Ginkgo   Time               " << ginkgo_time << " seconds." << std::endl;
        std::cout << " Ginkgo   Convergence status " << std::boolalpha << logger->has_converged() << "." << std::endl;
        std::cout << " Ginkgo   Iteration count    " << logger->get_num_iterations() << "." << std::endl;    
        auto residual_norm = gko::as<ginkgo_vec_type>(logger->get_residual_norm());
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
