
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

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include <cassert>
#include "GenerateNonsymCoarseProblem.hpp"
#include "GenerateNonsymProblem.hpp"
#include "SetupHalo.hpp"

/*!
  Routine to construct a prolongation/restriction operator for a given fine grid matrix
  solution (as computed by a direct solver).

  @param[inout]  Af - The known system matrix, on output its coarse operator, fine-to-coarse operator and auxiliary vectors will be defined.

  Note that the matrix Af is considered const because the attributes we are modifying are declared as mutable.

*/

template<class SparseMatrix_type>
void GenerateNonsymCoarseProblem(DeviceCtx* const dctx, const SparseMatrix_type& Af)
{

    // Use halo_scalar_type here since, for now, it represents the uniform precision of
    //   all working vectors inside a restart cycle.
    typedef typename SparseMatrix_type::halo_scalar_type scalar_type;
    typedef Vector<scalar_type> Vector_type;
    typedef MGData<scalar_type> MGData_type;

    // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
    // below may result in global range values.
    global_int_t nxf = Af.geom->nx;
    global_int_t nyf = Af.geom->ny;
    global_int_t nzf = Af.geom->nz;

    assert(nxf % 2 == 0);
    assert(nyf % 2 == 0);
    assert(nzf % 2 == 0); // Need fine grid dimensions to be divisible by 2
    const local_int_t nxc                 = nxf / 2;
    const local_int_t nyc                 = nyf / 2;
    const local_int_t nzc                 = nzf / 2;
    const local_int_t c_localNumberOfRows = nxc * nyc * nzc; // This is the size of our subblock
    // Throw an exception of the number of rows is less than zero (can happen if "int" overflows)
    // If this assert fails, it most likely means that the local_int_t is set to int
    //   and should be set to long long
    assert(c_localNumberOfRows > 0);
    local_int_t* f2cOperator = new local_int_t[c_localNumberOfRows];
    local_int_t* c2fOperator = new local_int_t[Af.localNumberOfRows];
    auto* d_f2cOperator      = static_cast<local_int_t*>(
        dctx->device_alloc(c_localNumberOfRows * sizeof(local_int_t)));
    auto* d_c2fOperator = static_cast<local_int_t*>(
        dctx->device_alloc(Af.localNumberOfRows * sizeof(local_int_t)));

    // Use a parallel loop to do initial assignment:
    // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPGMP_NO_OPENMP
    // clang-format off
    #pragma omp parallel for
    // clang-format on
#endif
    for (local_int_t i = 0; i < c_localNumberOfRows; ++i) {
        f2cOperator[i] = 0;
    }
    for (local_int_t i = 0; i < Af.localNumberOfRows; ++i) {
        c2fOperator[i] = -1;
    }


    // TODO:  This triply nested loop could be flattened or use nested parallelism
#ifndef HPGMP_NO_OPENMP
    // clang-format off
    #pragma omp parallel for
    // clang-format on
#endif
    for (local_int_t izc = 0; izc < nzc; ++izc) {
        local_int_t izf = 2 * izc;
        for (local_int_t iyc = 0; iyc < nyc; ++iyc) {
            local_int_t iyf = 2 * iyc;
            for (local_int_t ixc = 0; ixc < nxc; ++ixc) {
                local_int_t ixf               = 2 * ixc;
                local_int_t currentCoarseRow  = izc * nxc * nyc + iyc * nxc + ixc;
                local_int_t currentFineRow    = izf * nxf * nyf + iyf * nxf + ixf;
                f2cOperator[currentCoarseRow] = currentFineRow;
                c2fOperator[currentFineRow]   = currentCoarseRow;
            } // end iy loop
        } // end even iz if statement
    } // end iz loop

    // Copy to device
    dctx->copy_host_to_device_sync(d_f2cOperator, f2cOperator, c_localNumberOfRows * sizeof(local_int_t));
    dctx->copy_host_to_device_sync(d_c2fOperator, c2fOperator, Af.localNumberOfRows * sizeof(local_int_t));
    // host copy of c2f is not required.
    delete[] c2fOperator;
    // TODO: host copy of f2c is also not required for ELL RB solver.

    // Construct the geometry and linear system
    Geometry* geomc = new Geometry;
    local_int_t zlc = 0; // Coarsen nz for the lower block in the z processor dimension
    local_int_t zuc = 0; // Coarsen nz for the upper block in the z processor dimension
    int pz          = Af.geom->pz;
    if (pz > 0) {
        zlc = Af.geom->partz_nz[0] / 2; // Coarsen nz for the lower block in the z processor dimension
        zuc = Af.geom->partz_nz[1] / 2; // Coarsen nz for the upper block in the z processor dimension
    }
    GenerateGeometry(Af.geom->size, Af.geom->rank, Af.geom->numThreads, Af.geom->pz, zlc, zuc,
                     nxc, nyc, nzc, Af.geom->npx, Af.geom->npy, Af.geom->npz, geomc);

    const bool init_vect = false;
    Vector_type* tmp{};
    SparseMatrix_type* Ac = new SparseMatrix_type;
    Ac->initialize(geomc, Af.comm, dctx);
    GenerateNonsymProblem(dctx, *Ac, tmp, tmp, tmp, init_vect);
    SetupHalo(*Ac);
    Vector_type* rc     = new Vector_type(Ac->localNumberOfRows, Ac->comm, dctx);
    Vector_type* xc     = new Vector_type(Ac->localNumberOfColumns, Ac->comm, dctx);
    Vector_type* Axf    = new Vector_type(Af.localNumberOfColumns, Ac->comm, dctx);
    Af.Ac               = Ac;
    MGData_type* mgData = new MGData_type(dctx, f2cOperator, d_f2cOperator, d_c2fOperator,
                                          rc, xc, Axf);
    // NOTE: SparseMatrix Af takes ownership of mgData and deletes it when it's destroyed.
    Af.mgData = mgData;

    return;
}


/* --------------- *
 * specializations *
 * --------------- */

template void GenerateNonsymCoarseProblem< SparseMatrix<double> >(
    DeviceCtx*, SparseMatrix<double> const&);

template void GenerateNonsymCoarseProblem< SparseMatrix<float> >(
    DeviceCtx*, SparseMatrix<float> const&);

template void GenerateNonsymCoarseProblem< SparseMatrix<double, float> >(
    DeviceCtx*, SparseMatrix<double, float> const&);
