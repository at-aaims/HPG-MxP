
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
 @file ComputeMG.cpp

 HPGMP routine
 */

#include "ComputeMG.hpp"

#include <cassert>

#include "ell_multicolor_gs.hpp"
#include "restriction.hpp"
#include "prolongation.hpp"
#include "mytimer.hpp"

#ifdef HPGMP_WITH_GINKGO
#include "GinkgoOptData.hpp"
#endif

/*!
 * @param[in] A the known system matrix
 * @param[in] r the residual or input RHS on the finest grid.
 * @param[inout] x On exit contains the result of the multigrid V-cycle
 *                 x is the approximation to Ax = r.
 *                 Also contains timing information of the whole multigrid preconditioner on exit.
 * @param[inout] ft  It is updated with the flops, memory traffic etc. for the global multigrid
 *                   preconditioner.
 *
 * The total number of flops in the entire MG preconditioner with n levels is
 * 2*nnz_allgrids*(1 + n_presmooth + n_postsmooth) - 2*(nnz_finest - nnz_coarsest).
 * Note that this is cheaper than the reference ComputeMG_ref because of fusing the residual
 * calculation and restriction.
 *
 * @return returns 0 upon success and non-zero otherwise
 *
 * @see ComputeMG_ref
 */
template<class SparseMatrix_type, class Vector_type>
int ComputeMG(const SparseMatrix_type& A, const Vector_type& r, Vector_type& x,
              const bool symmetric, perf_counters& ft)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    using scalar_type       = typename SparseMatrix_type::scalar_type;
    using local_scalar_type = typename SparseMatrix_type::local_scalar_type;
    using halo_scalar_type  = typename SparseMatrix_type::halo_scalar_type;
    using vec_salar_type    = typename Vector_type::scalar_type;
    const int mpisize       = A.geom->size;

    // Optimized versions of calls
    double t0 = 0.0;
    x.fill_zero();
    //ft.mg_rp.f_mem_traffic[0] += x.local_length();
    ft.mg_rp.add_memory_traffic<vec_salar_type>(mpisize * x.local_length());

#ifdef HPGMP_WITH_GINKGO
    std::shared_ptr<const GinkgoMatrix<local_scalar_type, halo_scalar_type>> mat =
        dynamic_cast<GinkgoOptData<local_scalar_type, halo_scalar_type>*>(A.optimizationData)->mat;
    std::shared_ptr<const GinkgoSmoother<local_scalar_type, halo_scalar_type>> smoother =
        dynamic_cast<GinkgoOptData<local_scalar_type, halo_scalar_type>*>(A.optimizationData)->smoother;
#else
    std::shared_ptr<const ELLMatrix<local_scalar_type, halo_scalar_type>> mat =
        dynamic_cast<EllOptData<local_scalar_type, halo_scalar_type>*>(A.optimizationData)->mat;
#endif

    int ierr = 0;
    if (A.mgData != 0)
    {
        const int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;

        // Intentional abuse of Vector class timing variables
        auto time_gs_sofar = x.time2_;
        x.time2_           = 0.0;

        HPGMP_RANGE_PUSH("ell_multicolor_gs");

        TICK();

        for (int i = 0; i < numberOfPresmootherSteps; ++i) {
            if (i == 0) {
#ifdef HPGMP_WITH_GINKGO
                ierr += ginkgo_multicolor_gs(smoother.get(), mat.get(), &r, &x);
#else
                ierr += ell_multicolor_gs_zero_initial(symmetric, mat.get(), &r, &x);
#endif
                //ft.mg_gs.flops[0] += A.totalNumberOfNonzeros;
                ft.mg_gs.add_flops<scalar_type>(A.totalNumberOfNonzeros);
                // the first color is treated differently:
                ft.mg_gs.add_memory_traffic<scalar_type>(
                    3.0 * A.totalNumberOfRows / 8 + //
                    7.0 / 8 * (A.totalNumberOfNonzeros + A.totalNumberOfRows - 1) / 2 + //
                    1 + 2 * A.totalNumberOfRows);
                ft.mg_gs.add_memory_traffic<int>(7.0 / 8 * A.totalNumberOfNonzeros);
            } else {
#ifdef HPGMP_WITH_GINKGO
                ierr += ginkgo_multicolor_gs(smoother.get(), mat.get(), &r, &x);
#else
                ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
#endif
                //ft.mg_gs.flops[0] += 2*A.totalNumberOfNonzeros;
                ft.mg_gs.add_flops<scalar_type>(2 * A.totalNumberOfNonzeros);
                ft.mg_gs.add_memory_traffic<scalar_type>(A.totalNumberOfNonzeros + 2 * A.totalNumberOfRows);
                ft.mg_gs.add_memory_traffic<int>(A.totalNumberOfNonzeros);
            }
        }
        x.time1_ = 0.0; // Apparently no one cares about GS comm time
        // restore GS time so far
        x.time2_ = time_gs_sofar;
        TOCK(x.time2_);

        HPGMP_RANGE_POP("ell_multicolor_gs");
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }

        // Restriction operation
        HPGMP_RANGE_PUSH("fused_spmv_restriction");
        TICK();
        //ierr = restriction(A, r);
        ierr = fused_spmv_restriction(A, r, x);
        TOCK(x.time3_);
        //ft.mg_rp.flops[0] += 2*A.Ac->totalNumberOfNonzeros;
        const auto coarse_len = A.Ac->totalNumberOfRows;
        ft.mg_rp.add_flops<scalar_type>(2 * A.Ac->totalNumberOfNonzeros);
        ft.mg_rp.add_memory_traffic<local_int_t>(3 * coarse_len + A.Ac->totalNumberOfNonzeros);
        ft.mg_rp.add_memory_traffic<scalar_type>(coarse_len + A.Ac->totalNumberOfNonzeros + A.totalNumberOfRows);
        HPGMP_RANGE_POP("fused_spmv_restriction");
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }

        // MG on coarser-grid
        A.mgData->xc->time1_ = A.mgData->xc->time2_ = 0.0;
        A.mgData->xc->time3_ = A.mgData->xc->time4_ = 0.0;
        ierr                                        = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc, symmetric, ft);
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
        x.time1_ += A.mgData->xc->time1_;
        x.time2_ += A.mgData->xc->time2_;
        x.time3_ += A.mgData->xc->time3_;
        x.time4_ += A.mgData->xc->time4_;

        // Prolongation operation
        HPGMP_RANGE_PUSH("prolongation");
        TICK();
        ierr = prolongation(A, x);
        TOCK(x.time4_);
        //ft.mg_rp.flops[0] += mpisize*A.mgData->rc->local_length();
        ft.mg_rp.add_flops<scalar_type>(coarse_len);
        ft.mg_rp.add_memory_traffic<local_int_t>(coarse_len * 3);
        ft.mg_rp.add_memory_traffic<scalar_type>(coarse_len * 2);
        HPGMP_RANGE_POP("prolongation");
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }

        // Post-smoothing
        time_gs_sofar = x.time2_;
        x.time2_      = 0.0;
        HPGMP_RANGE_PUSH("post-smoothing");
        TICK();

        const int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
        for (int i = 0; i < numberOfPostsmootherSteps; ++i) {
#ifdef HPGMP_WITH_GINKGO
            ierr += ginkgo_multicolor_gs(smoother.get(), mat.get(), &r, &x);
#else
            ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
#endif
            //ft.mg_gs.flops[0] += 2*A.totalNumberOfNonzeros;
            ft.mg_gs.add_flops<scalar_type>(2 * A.totalNumberOfNonzeros);
            ft.mg_gs.add_memory_traffic<scalar_type>(A.totalNumberOfNonzeros + 2 * A.totalNumberOfRows);
            ft.mg_gs.add_memory_traffic<local_int_t>(A.totalNumberOfNonzeros);
        }
        x.time1_ = 0.0; // Apparently no one cares about GS comm time
        // restore GS time so far
        x.time2_ = time_gs_sofar;
        TOCK(x.time2_);
        HPGMP_RANGE_POP("post-smoothing");
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
    } else {
        // coarsest grid
        auto time_gs_sofar = x.time2_;
        x.time2_           = 0.0;
        HPGMP_RANGE_PUSH("ell_multicolor_gs");
        TICK();
#ifdef HPGMP_WITH_GINKGO
        ierr += ginkgo_multicolor_gs(smoother.get(), mat.get(), &r, &x);
#else
        ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
#endif

        x.time1_ = 0.0; // Apparently no one cares about GS comm time
        // restore GS time so far
        x.time2_ = time_gs_sofar;
        TOCK(x.time2_);
        HPGMP_RANGE_POP("ell_multicolor_gs");
        //ft.mg_gs.flops[0] += 2*A.totalNumberOfNonzeros;
        ft.mg_gs.add_flops<scalar_type>(2 * A.totalNumberOfNonzeros);
        ft.mg_gs.add_memory_traffic<scalar_type>(A.totalNumberOfNonzeros + 2 * A.totalNumberOfRows);
        ft.mg_gs.add_memory_traffic<local_int_t>(A.totalNumberOfNonzeros);
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
    }

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeMG(
    SparseMatrix<double> const&, Vector<double> const&, Vector<double>&, bool, perf_counters&);

template int ComputeMG(
    SparseMatrix<float> const&, Vector<float> const&, Vector<float>&, bool, perf_counters&);

#ifdef HPGMP_WITH_GINKGO_AMP
template int ComputeMG(
    SparseMatrix<double, float> const&, Vector<float> const&, Vector<float>&, bool, perf_counters&);
template int ComputeMG(
    SparseMatrix<double, float> const&, Vector<double> const&, Vector<double>&, bool, perf_counters&);
#endif
