
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
 @file CGData.hpp

 HPGMP data structure
 */

#ifndef GMRESDATA_HPP
#define GMRESDATA_HPP

#include <array>
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "perf_counter.hpp"

template<typename local_scalar_t, typename halo_scalar_t, typename project_scalar_t>
class GMRESData
{
public:
    using local_scalar_type = local_scalar_t;
    using halo_scalar_type  = halo_scalar_t;
    using project_type      = project_scalar_t;

    /*!
   * Constructor for the data structure of GMRES vectors.
   *
   * @param[in]  A    the data structure that describes the problem matrix and its structure
   */
    GMRESData(SparseMatrix<local_scalar_t, halo_scalar_t>& A, DeviceCtx* const dctx)
        : r(A.localNumberOfRows, A.comm, dctx),
          z(A.localNumberOfColumns, A.comm, dctx),
          p(A.localNumberOfColumns, A.comm, dctx),
          w(A.localNumberOfRows, A.comm, dctx),
          Ap(A.localNumberOfRows, A.comm, dctx)
    { }

    GMRESData() { }

    void initialize(SparseMatrix<local_scalar_t, halo_scalar_t>& A, DeviceCtx* const dctx)
    {
        r.initialize(A.localNumberOfRows, A.comm, dctx);
        z.initialize(A.localNumberOfColumns, A.comm, dctx);
        p.initialize(A.localNumberOfColumns, A.comm, dctx);
        w.initialize(A.localNumberOfRows, A.comm, dctx);
        Ap.initialize(A.localNumberOfRows, A.comm, dctx);
    }

    Vector<halo_scalar_t> r; //!< pointer to residual vector
    Vector<halo_scalar_t> z; //!< pointer to preconditioned residual vector
    Vector<halo_scalar_t> p; //!< pointer to direction vector
    Vector<halo_scalar_t> w; //!< pointer to workspace
    Vector<halo_scalar_t> Ap; //!< pointer to Krylov vector
};


class TestGMRESData
{
public:
    static constexpr int n_fl_ops    = 4; ///< Number of numerical operations for which flops are counted in GMRES
    static constexpr int n_timed_ops = 12; ///< Number of operations which are timed in GMRES

    int restart_length; //!< restart length
    double tolerance; //!< tolerance = reference residual norm
    double runningTime; //!<
    double minOfficialTime; //!<

    // from validation step
    int refNumIters; //!< number of reference iterations
    int optNumIters; //!< number of optimized iterations
    int validation_nprocs; //!<
    double refResNorm0;
    double refResNorm;
    double optResNorm0;
    double optResNorm;

    // setup time
    double SetupTime;
    double OptimizeTime;
    double SpmvMgTime;

    // from benchmark step
    int numOfCalls; //!< number of calls
    int maxNumIters; //!<
    int numOfMGCalls; //!< number of MG calls
    int numOfSPCalls; //!< number of SpMV calls
    int optNumOfMGCalls; //!< number of MG calls   (opt)
    int optNumOfSPCalls; //!< number of SpMV calls (opt)
    int refNumOfMGCalls; //!< number of MG calls   (ref)
    int refNumOfSPCalls; //!< number of SpMV calls (ref)
    double refTotalFlops; //
    double refTotalTime; //
    double optTotalFlops; //
    double optTotalTime; //

    perf_counters ctrs_ref; ///< Counting for flops and memory traffic
    perf_counters ctrs_bench; ///< Counting for flops and memory traffic of different precisions

    //! flop counts and time for total, dot, axpy, ortho, spmv, reduce, precond
    std::array<double, n_fl_ops> flops; //!< accumulated in GMRES, temporary workspace
    std::array<double, n_timed_ops> times; //!< accumulated in GMRES, temporary workspace
    std::array<double, n_timed_ops> times_comp; //!< accumulated in GMRES, temporary workspace
    std::array<double, n_timed_ops> times_comm; //!< accumulated in GMRES, temporary workspace

    std::array<double, n_fl_ops> ref_flops; //!< record from output of reference GMRES
    std::array<double, n_timed_ops> ref_times; //!< record from output of reference GMRES
    std::array<double, n_fl_ops> opt_flops; //!< record from output of optimized GMRES
    std::array<double, n_timed_ops> opt_times; //!< record from output of optimized GMRES
    std::array<double, n_timed_ops> ref_times_comp; //!< record from output of reference GMRES
    std::array<double, n_timed_ops> ref_times_comm; //!< record from output of reference GMRES
    std::array<double, n_timed_ops> opt_times_comp; //!< record from output of optimized GMRES
    std::array<double, n_timed_ops> opt_times_comm; //!< record from output of optimized GMRES
};

#endif // CGDATA_HPP
