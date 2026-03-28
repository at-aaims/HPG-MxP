
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

    // Call user-tunable set up function.
    double t7 = mytimer();
    OptimizeProblem(A, data, b, x, xexact);
    t7       = mytimer() - t7;
    times[7] = t7;

#ifdef HPGMP_VERBOSE
    {
        std::cout << "A.localNumberOfRows: " << A.localNumberOfRows << "\n";
        std::cout << "A.localNumberOfColumns: " << A.localNumberOfColumns << "\n";
        std::cout << "A.localNumberOfNonzeros: " << A.localNumberOfNonzeros << "\n";
        std::cout << "A.colInd: (First 10 rows)\n";
        for (local_int_t row = 0; row < 10; ++row)
        {
            auto currentRowColInd = A.mtxIndL[row];
            for (local_int_t col = 0; col < A.nonzerosInRow[row]; ++col)
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
            for (local_int_t col = 0; col < A.nonzerosInRow[row]; ++col)
            {
                std::cout << currentRowValues[col] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "b.values(): (First 10 rows)\n";
        auto b_values = b.values();
        for (local_int_t row = 0; row < 10; ++row)
        {
            std::cout << b_values[row] << "\n";
        }
        std::cout << "xexact.values(): (First 10 rows)\n";
        auto xexact_values = xexact.values();
        for (local_int_t row = 0; row < 10; ++row)
        {
            std::cout << xexact_values[row] << "\n";
        }
    }
#endif // HPGMP_VERBOSE

    // Use Ginkgo to solve
    double gko_time    = mytimer();
    using gmres        = gko::solver::Gmres<>;
    using bj           = gko::preconditioner::Jacobi<>;
    using gko_vec_type = gko::matrix::Dense<scalar_type>;
    using gko_ell_type = gko::matrix::Ell<scalar_type, local_int_t>;
    using gko_coo_type = gko::matrix::Coo<scalar_type, local_int_t>;
    auto gko_exec      = create_ginkgo_executor();
#ifdef HPGMP_REFERENCE
    using gko_mat_type = gko_coo_type;
    auto gko_mat =
        gko::share(gko_mat_type::create(gko_exec,
                                        gko::dim<2>{static_cast<gko::size_type>(A.localNumberOfRows),
                                                    static_cast<gko::size_type>(A.localNumberOfColumns)},
                                        A.localNumberOfNonzeros));
    auto rhs =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(A.localNumberOfRows), 1},
                             1);
    auto u =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(A.localNumberOfRows), 1},
                             1);
    auto gko_mat_rows   = gko_mat->get_row_idxs();
    auto gko_mat_cols   = gko_mat->get_col_idxs();
    auto gko_mat_values = gko_mat->get_values();
    auto rhs_values     = rhs->get_values();
    auto u_values       = u->get_values();
    auto b_values       = b.values();
    auto x_values       = x.values();
    local_int_t i       = 0;
    for (local_int_t row = 0; row < A.localNumberOfColumns; ++row)
    {
        auto currentRowValues = A.matrixValues[row];
        auto currentRowColInd = A.mtxIndL[row];
        for (local_int_t col = 0; col < A.nonzerosInRow[row]; ++col)
        {
            if (currentRowColInd[col] != -1)
            {
                gko_mat_rows[i]   = row;
                gko_mat_cols[i]   = currentRowColInd[col];
                gko_mat_values[i] = currentRowValues[col];
                i++;
            }
        }
        rhs_values[row] = b_values[row];
        u_values[row]   = x_values[row];
    }
#else // HPGMP_REFERENCE
    using gko_mat_type = gko_ell_type;
    std::shared_ptr<const ELLMatrix<scalar_type>> mat =
        dynamic_cast<EllOptData<scalar_type>*>(A.optimizationData)->mat;
    auto mat_ptr = mat.get();
#ifdef HPGMP_VERBOSE
    {
        size_t mat_values_bytes       = mat_ptr->get_ld_values() * mat_ptr->get_ell_width() * sizeof(scalar_type);
        size_t mat_col_bytes          = mat_ptr->get_ld_indices() * mat_ptr->get_ell_width() * sizeof(local_int_t);
        scalar_type* h_tmp_mat_values = (scalar_type*)malloc(mat_values_bytes);
        local_int_t* h_tmp_col_values = (local_int_t*)malloc(mat_col_bytes);
        dctx.get()->copy_device_to_host_sync((void*)h_tmp_mat_values, mat_ptr->get_values(), mat_values_bytes);
        dctx.get()->copy_device_to_host_sync((void*)h_tmp_col_values, mat_ptr->get_col_idxs(), mat_col_bytes);
        std::cout << "ELL matrix column indices (copied to host): (First 20 rows)\n";
        for (local_int_t row = 0; row < 20; ++row)
        {
            for (local_int_t col = 0; col < mat_ptr->get_ell_width(); ++col)
            {
                std::cout << h_tmp_col_values[row * mat_ptr->get_ell_width() + col] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "ELL matrix values (copied to host): (First 20 rows)\n";
        for (local_int_t row = 0; row < 20; ++row)
        {
            for (local_int_t col = 0; col < mat_ptr->get_ell_width(); ++col)
            {
                std::cout << h_tmp_mat_values[row * mat_ptr->get_ell_width() + col] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "Permutted rhs for ELL (updated host mirror): (First 20 rows)\n";
        auto b_values = b.values();
        for (local_int_t row = 0; row < 20; ++row)
        {
            std::cout << b_values[row] << "\n";
        }
        free(h_tmp_mat_values);
        free(h_tmp_col_values);
    }
#endif // HPGMP_VERBOSE
    auto gko_mat =
        gko::share(gko_mat_type::create_const(gko_exec,
                                              gko::dim<2>{static_cast<gko::size_type>(mat_ptr->get_local_num_rows()),
                                                          static_cast<gko::size_type>(mat_ptr->get_local_num_cols())},
                                              gko::make_const_array_view(gko_exec,
                                                                         mat_ptr->get_ld_values() * mat_ptr->get_ell_width(),
                                                                         mat_ptr->get_values()),
                                              gko::make_const_array_view(gko_exec,
                                                                         mat_ptr->get_ld_indices() * mat_ptr->get_ell_width(),
                                                                         mat_ptr->get_col_idxs()),
                                              mat_ptr->get_ell_width(),
                                              mat_ptr->get_ld_values()));
    auto rhs =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(b.local_length()), 1},
                             gko::make_array_view(gko_exec,
                                                  b.local_length(),
                                                  b.d_values()),
                             1);
    auto u =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(x.local_length()), 1},
                             gko::make_array_view(gko_exec,
                                                  x.local_length(),
                                                  x.d_values()),
                             1);

#endif // HPGMP_REFERENCE

    auto solver_factory = gmres::build()
                              .with_criteria(gko::stop::Iteration::build()
                                                 .with_max_iters(1000)
                                                 .on(gko_exec),
                                             gko::stop::ResidualNorm<scalar_type>::build()
                                                 .with_reduction_factor(1e-9)
                                                 .on(gko_exec))
                              .with_preconditioner(bj::build().with_max_block_size(8u).on(gko_exec))
                              .on(gko_exec);

    auto solver = solver_factory->generate(gko_mat);

    std::shared_ptr<const gko::log::Convergence<scalar_type>> logger = gko::log::Convergence<scalar_type>::create();

    solver->add_logger(logger);
    solver->apply(rhs, u);
    gko_time = mytimer() - gko_time;

#ifdef HPGMP_VERBOSE
    {
        auto u_values = u->get_values();
        std::cout << "u.values(): (First 10 rows)\n";
        for (local_int_t i = 0; i < 10; ++i)
        {
            std::cout << u_values[i] << "\n";
        }
    }
#endif

    if (A.geom->rank == 0) {
        std::cout << " Setup    Time               " << setup_time << " seconds." << std::endl;
        std::cout << " Optimize Time               " << t7 << " seconds." << std::endl;
        std::cout << " Ginkgo   Time               " << gko_time << " seconds." << std::endl;
        std::cout << " Ginkgo   Convergence status " << std::boolalpha << logger->has_converged() << "." << std::endl;
        std::cout << " Ginkgo   Iteration count    " << logger->get_num_iterations() << "." << std::endl;
        auto residual_norm = gko::as<gko_vec_type>(logger->get_residual_norm());
        std::cout << " Ginkgo   Residual norm      " << residual_norm->at(0, 0) << "." << std::endl;
    }

    // check status
    int status = 0;
    if (!logger->has_converged()) {
        status = 1;
    }

    // free
    DeleteMatrix(A);
    delete geom;
    HPGMP_Finalize();
#ifndef HPGMP_NO_MPI
    MPI_Finalize();
#endif
    return status;
}
