#undef NDEBUG
#include <cstdlib>
#include <memory>
#include <vector>
#include <set>
#include <utility>

#include "CheckAspectRatio.hpp"
#include "CheckProblem.hpp"
#include "mytimer.hpp"
#include "GMRESData.hpp"
#include "SetupMatrix.hpp"
#include "multicoloring.hpp"

#include "testbase.hpp"

using scalar      = float;
using vector_t    = Vector<scalar>;
using spMatrix_t  = SparseMatrix<scalar>;
using gmresData_t = GMRESData<scalar>;

int run_all_tests(comm_type comm, const HPGMP_Params& params)
{
    int ierr = 0;

    const int size = params.comm_size, rank = params.comm_rank;
    auto dctx = std::make_unique<DeviceCtx>(rank);

    const local_int_t nx = 16;
    const local_int_t ny = 16;
    const local_int_t nz = 16;

    ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank == 0);
    if (ierr)
        return ierr;

    // Construct the geometry
    Geometry* geom = new Geometry;
    GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz,
                     params.npx, params.npy, params.npz, geom);

    ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank == 0);
    if (ierr)
        return ierr;

    spMatrix_t A;

    const bool init_vect = true;
    vector_t b, x, xexact;
    gmresData_t data;

    // Number of levels including first
    const int numberOfMgLevels = 1;
    // setup matrix
    SetupMatrix(dctx.get(), numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect, comm);

    const auto local_nrows = A.localNumberOfRows;

    // allocate and copy extra arrays needed by coloring
    A.d_mtxIndL = reinterpret_cast<local_int_t*>(
        dctx->device_alloc(A.max_nnz_per_row * local_nrows * sizeof(local_int_t)));
    dctx->copy_host_to_device_sync(A.d_mtxIndL, A.mtxIndL[0],
                                   A.max_nnz_per_row * local_nrows * sizeof(local_int_t));
    A.d_matrixValues = reinterpret_cast<scalar*>(
        dctx->device_alloc(A.max_nnz_per_row * local_nrows * sizeof(scalar)));
    dctx->copy_host_to_device_sync(A.d_matrixValues, A.matrixValues[0],
                                   A.max_nnz_per_row * local_nrows * sizeof(scalar));

    multicolor_JPL(A);
    //multicolor_ref(A);

    assert(A.nblocks == 8);

    for (int i = 0; i < A.nblocks; i++) {
        printf("Npoin in color %d is %d.\n", i, A.sizes[i]);
        assert(A.sizes[i] == A.offsets[i + 1] - A.offsets[i]);
    }

    auto host_perm = new local_int_t[local_nrows];
    auto row_color = new int[local_nrows];

    dctx->copy_device_to_host_sync(host_perm, A.perm, local_nrows * sizeof(local_int_t));

    for (int ic = 0; ic < A.nblocks; ic++) {
        const int start = A.offsets[ic];
        const int end   = A.offsets[ic + 1];
        for (int irow = 0; irow < A.localNumberOfRows; irow++) {
            if (host_perm[irow] < end && host_perm[irow] >= start) {
                row_color[host_perm[irow]] = ic;
            }
        }
    }

    std::vector<std::pair<int, int>> violations;
    std::set<int> bad_rows;

    for (int irow = 0; irow < local_nrows; irow++) {
        const int icolor = row_color[host_perm[irow]];
        bool bad         = false;
        for (int j = 0; j < A.nonzerosInRow[irow]; j++) {
            const int col = A.mtxIndL[irow][j];
            if ((col < local_nrows)) {
                if (irow != col) {
                    const int jcolor = row_color[host_perm[col]];
                    // If neighbor has the same color, flag it
                    if (jcolor == icolor) {
                        violations.push_back(std::pair{irow, col});
                        bad = true;
                    }
                }
            }
        }
        if (bad) {
            bad_rows.insert(irow);
        }
    }

    printf("%lu violations, %lu bad rows\n  row, col\n", violations.size(), bad_rows.size());
    if (violations.size() > 0) {
        auto vit = violations.begin();
        for (int i = 0; i < 3; ++vit, ++i) {
            printf(" %d,  %d\n", std::get<0>(*vit), std::get<1>(*vit));
        }
        fflush(stdout);
    }
    assert(violations.size() == 0);

    delete[] host_perm;
    delete[] row_color;
    dctx->device_free(A.d_mtxIndL);
    dctx->device_free(A.d_matrixValues);
    //DeleteMatrix(A);
    delete geom;
    return ierr;
}
