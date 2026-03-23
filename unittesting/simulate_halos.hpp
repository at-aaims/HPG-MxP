#ifndef HPGMP_UNITTESTING_SIMULATE_HALOS_HPP
#define HPGMP_UNITTESTING_SIMULATE_HALOS_HPP

//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// ***************************************************
//@HEADER

#include "Geometry.hpp"
#include "SparseMatrix.hpp"

template<class SparseMatrix_type>
local_int_t simulate_halos(SparseMatrix_type& A)
{
    const int max_nnz_per_row = 27;
    const local_int_t nrows   = A.localNumberOfRows;
    A.numberOfSendNeighbors   = 1;
    auto oldmtxIndL           = A.mtxIndL[0];
    auto nnz_row              = A.nonzerosInRow;
    auto nonzerosInRow        = A.nonzerosInRow;
    const Geometry* const g   = A.geom;
    const std::array<int, 3> ln{g->nz, g->ny, g->nx};
    local_int_t ihalo         = nrows;
    local_int_t num_halo_rows = 0;
    for (local_int_t i = 0; i < nrows; i++) {
        const auto cur_nnz = static_cast<local_int_t>(nonzerosInRow[i]);
        const auto p       = get_local_3d_from_flattened(i, ln);
        // Add one ghost cell per halo cell
        if (p[0] == 0 || p[1] == 0 || p[2] == 0 ||
            p[0] == g->nz - 1 || p[1] == g->ny - 1 || p[2] == g->nx - 1)
        {
            assert(cur_nnz < max_nnz_per_row);
            A.mtxIndL[i][cur_nnz] = ihalo++;
            num_halo_rows++;
        }
        bool corner = true;
        for (int idir = 0; idir < 3; idir++) {
            if (p[idir] != 0 && p[idir] != ln[idir] - 1) {
                corner = false;
            }
        }
        // Add 19 extra at corners
        const int nhalocorners = 19;
        if (corner) {
            assert(cur_nnz + nhalocorners <= max_nnz_per_row);
            for (int ic = 0; ic < nhalocorners; ic++) {
                A.mtxIndL[i][cur_nnz + ic] = ihalo++;
            }
        }
        // We ignore extra halo dependencies at edges.
    }
    const local_int_t num_external_nz = ihalo - nrows;
    A.numberOfExternalValues = A.totalToBeSent = num_external_nz;
    if (A.elementsToSend) {
        delete[] A.elementsToSend;
    }
    A.elementsToSend            = new local_int_t[A.totalToBeSent];
    const int num_external_dofs = 2 * (ln[0] * ln[1] + ln[1] * ln[2] + ln[2] * ln[0]) + 8 * 4;
    A.localNumberOfColumns      = nrows + num_external_dofs;
    printf("simulate_halos: Added %d halo DOFs and %d halo dependencies.\n", num_external_dofs,
           num_external_nz);
    return num_halo_rows;
}

#endif
