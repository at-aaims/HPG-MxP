
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
 @file TestGMRES.cpp

 HPGMP routine
 */

// Changelog
// None
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
using std::endl;
#include <vector>
#include "hpgmp.hpp"

#include "TestGMRES.hpp"
#include "mytimer.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"

/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
template<class Vector_type>
inline void ScaleVectorValue(Vector_type& v, local_int_t index, typename Vector_type::scalar_type value)
{
    typedef typename Vector_type::scalar_type scalar_type;
    assert(index >= 0 && index < v.local_length());
    scalar_type* vv = v.values();
    vv[index] *= value;
    return;
}

/*!
  Test the correctness of the Preconditined CG implementation by using a system matrix with a dominant diagonal.

  @param[in]    geom The description of the problem's geometry.
  @param[in]    A    The known system matrix
  @param[in]    data the data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution

  @return Returns zero on success and a non-zero value otherwise.

  @see CG()
 */


template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type>
int TestGMRES(SparseMatrix_type& A, SparseMatrix_type2& A_lo, GMRESData_type& data, GMRESData_type2& data_lo,
              Vector_type& b, Vector_type& x, bool test_diagonal_exaggeration, bool test_noprecond, TestGMRESData& test_data)
{

    typedef typename SparseMatrix_type::scalar_type scalar_type;
    typedef typename SparseMatrix_type2::scalar_type scalar_type2;
    typedef Vector<scalar_type2> Vector_type2;

    // Use this array for collecting timing information
    // Temporary storage for holding original diagonal and RHS
    Vector_type origDiagA(A.localNumberOfRows, A.comm, x.get_device_context()),
        exaggeratedDiagA(A.localNumberOfRows, A.comm, x.get_device_context()),
        origB(A.localNumberOfRows, A.comm, x.get_device_context());
    Vector_type2 origDiagA2(A_lo.localNumberOfRows, A.comm, x.get_device_context()),
        exagDiagA2(A_lo.localNumberOfRows, A.comm, x.get_device_context());
    //InitializeVector(origDiagA, A.localNumberOfRows, A.comm);
    //InitializeVector(origDiagA2, A_lo.localNumberOfRows, A.comm);
    //InitializeVector(exaggeratedDiagA, A.localNumberOfRows, A.comm);
    //InitializeVector(exagDiagA2, A_lo.localNumberOfRows, A.comm);
    //InitializeVector(origB, A.localNumberOfRows, A.comm);
    CopyMatrixDiagonal(A, origDiagA);
    CopyMatrixDiagonal(A_lo, origDiagA2);
    CopyVector(origDiagA, exaggeratedDiagA);
    CopyVector(origDiagA2, exagDiagA2);
    CopyVector(b, origB);

    constexpr float scale_factor = 1.0e6;

    // TODO: This should be moved to somewhere-else, e.g., SetupProblem
    if (test_diagonal_exaggeration) {
        // Modify the matrix diagonal to greatly exaggerate diagonal values.
        // CG should converge in about 10 iterations for this problem, regardless of problem size
        if (A.geom->rank == 0) HPGMP_fout << std::endl
                                          << " ** applying diagonal exaggeration ** " << std::endl
                                          << std::endl;
        for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
            global_int_t globalRowID = A.localToGlobalMap[i];
            if (globalRowID < 9) {
                scalar_type scale   = (globalRowID + 2) * scale_factor;
                scalar_type2 scale2 = (globalRowID + 2) * scale_factor;
                ScaleVectorValue(exaggeratedDiagA, i, scale);
                ScaleVectorValue(exagDiagA2, i, scale2);
                ScaleVectorValue(b, i, scale);
            } else {
                ScaleVectorValue(exaggeratedDiagA, i, scale_factor);
                ScaleVectorValue(exagDiagA2, i, scale_factor);
                ScaleVectorValue(b, i, scale_factor);
            }
        }
        ReplaceMatrixDiagonal(A, exaggeratedDiagA);
        ReplaceMatrixDiagonal(A_lo, exagDiagA2); //TODO probably some funny casting here... need to do properly.
    } else {
        if (A.geom->rank == 0) HPGMP_fout << std::endl
                                          << " ** skippping diagonal exaggeration ** " << std::endl
                                          << std::endl;
    }

    int niters = 0;
    scalar_type normr(0.0);
    scalar_type normr0(0.0);
    int restart_length     = 40;
    int maxIters           = 10000;
    int numberOfGmresCalls = 1;
    bool verbose           = true;
    scalar_type tolerance  = 1.0e-12; // Set tolerance to reasonable value for grossly scaled diagonal terms

    constexpr int num_flops = TestGMRESData::n_fl_ops;
    constexpr int num_times = TestGMRESData::n_timed_ops;
    for (int i = 0; i < num_flops; i++) test_data.flops[i] = 0.0;
    for (int i = 0; i < num_times; i++) test_data.times[i] = 0.0;
    for (int i = 0; i < num_times; i++) test_data.times_comp[i] = 0.0;
    for (int i = 0; i < num_times; i++) test_data.times_comm[i] = 0.0;
    for (int k = (test_noprecond ? 0 : 1); k < 2; ++k)
    { // This loop tests both unpreconditioned and preconditioned runs
        for (int i = 0; i < numberOfGmresCalls; ++i) {
            x.fill_zero(); // Zero out x

            if (A.geom->rank == 0) {
                HPGMP_fout << "Calling GMRES (all double) for testing: " << endl;
            }
            double flops      = test_data.flops[0];
            double time_tic   = mytimer();
            int ierr          = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, k == 1, verbose, test_data);
            double time_solve = mytimer() - time_tic;
            flops             = test_data.flops[0] - flops;
            if (ierr) HPGMP_fout << "Error in call to GMRES: " << ierr << ".\n"
                                 << endl;
            if (A.geom->rank == 0) {
                HPGMP_fout << " [" << i << "] Number of GMRES Iterations [" << niters << "] Scaled Residual [" << normr / normr0 << "]" << endl;
                HPGMP_fout << " Time     " << time_solve << " seconds." << endl;
                HPGMP_fout << " Gflop/s  " << flops / 1000000000.0 << "/" << time_solve << " = " << (flops / 1000000000.0) / time_solve
                           << " (n = " << A.totalNumberOfRows << ")" << endl;
                HPGMP_fout << " Time/itr " << time_solve / niters << endl;
                if (normr / normr0 <= tolerance) {
                    HPGMP_fout << " ** PASS (normr = " << normr << " / " << normr0 << " = " << normr / normr0 << ", tol = " << tolerance << ") ** " << endl;
                } else {
                    HPGMP_fout << " ** FAIL (normr = " << normr << " / " << normr0 << " = " << normr / normr0 << ", tol = " << tolerance << ") ** " << endl;
                }
            }
        }
    }

#if 1
    for (int i = 0; i < num_flops; i++) test_data.flops[i] = 0.0;
    for (int i = 0; i < num_times; i++) test_data.times[i] = 0.0;
    for (int i = 0; i < num_times; i++) test_data.times_comp[i] = 0.0;
    for (int i = 0; i < num_times; i++) test_data.times_comm[i] = 0.0;
    for (int k = (test_noprecond ? 0 : 1); k < 2; ++k)
    { // This loop tests both unpreconditioned and preconditioned runs
        for (int i = 0; i < numberOfGmresCalls; ++i) {
            x.fill_zero(); // Zero out x

            if (A.geom->rank == 0) {
                HPGMP_fout << "Calling GMRES-IR for testing: " << endl;
            }
            double flops      = test_data.flops[0];
            double time_tic   = mytimer();
            int ierr          = GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, k, verbose, test_data);
            double time_solve = mytimer() - time_tic;
            flops             = test_data.flops[0] - flops;
            if (ierr) HPGMP_fout << "Error in call to GMRES-IR: " << ierr << ".\n"
                                 << endl;
            if (A.geom->rank == 0) {
                HPGMP_fout << "Call [" << i << "] Number of GMRES-IR Iterations [" << niters << "] Scaled Residual [" << normr / normr0 << "]" << endl;
                HPGMP_fout << " Time     " << time_solve << " seconds." << endl;
                HPGMP_fout << " Gflop/s  " << flops / 1000000000.0 << "/" << time_solve << " = " << (flops / 1000000000.0) / time_solve
                           << " (n = " << A.totalNumberOfRows << ")" << endl;
                HPGMP_fout << " Time/itr " << time_solve / niters << endl;
                if (normr / normr0 <= tolerance) {
                    HPGMP_fout << " ** PASS (normr = " << normr << " / " << normr0 << " = " << normr / normr0 << ", tol = " << tolerance << ") ** " << endl;
                } else {
                    HPGMP_fout << " ** FAIL (normr = " << normr << " / " << normr0 << " = " << normr / normr0 << ", tol = " << tolerance << ") ** " << endl;
                }
            }
        }
    }
#endif
    // Restore matrix diagonal and RHS
    ReplaceMatrixDiagonal(A, origDiagA);
    ReplaceMatrixDiagonal(A_lo, origDiagA2); //TODO again, probably funny casting here.
    CopyVector(origB, b);

    return 0;
}

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
int TestGMRES(SparseMatrix_type& A, GMRESData_type& data, Vector_type& b, Vector_type& x,
              bool test_diagonal_exaggeration, bool test_noprecond, TestGMRESData& test_data)
{
    return TestGMRES(A, A, data, data, b, x, test_diagonal_exaggeration, test_noprecond, test_data);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template int TestGMRES< SparseMatrix<double>,
                        GMRESData<double, double, double>,
                        Vector<double>>(
    SparseMatrix<double>&, GMRESData<double, double, double>&, Vector<double>&, Vector<double>&, bool, bool, TestGMRESData&);

template int TestGMRES< SparseMatrix<float>,
                        GMRESData<float, float, float>,
                        Vector<float>>(
    SparseMatrix<float>&, GMRESData<float, float, float>&, Vector<float>&, Vector<float>&, bool, bool, TestGMRESData&);


// uniform version
template int TestGMRES< SparseMatrix<double>,
                        SparseMatrix<double>,
                        GMRESData<double, double, double>,
                        GMRESData<double, double, double>,
                        Vector<double>>(
    SparseMatrix<double>&, SparseMatrix<double>&,
    GMRESData<double, double, double>&, GMRESData<double, double, double>&,
    Vector<double>&, Vector<double>&, bool, bool, TestGMRESData&);

template int TestGMRES< SparseMatrix<float>,
                        SparseMatrix<float>,
                        GMRESData<float, float, float>,
                        GMRESData<float, float, float>,
                        Vector<float>>(
    SparseMatrix<float>&, SparseMatrix<float>&,
    GMRESData<float, float, float>&, GMRESData<float, float, float>&,
    Vector<float>&, Vector<float>&, bool, bool, TestGMRESData&);


// mixed version
#ifdef HPGMP_WITH_GINKGO_AMP
template int TestGMRES< SparseMatrix<double>,
                        SparseMatrix<double, float>,
                        GMRESData<double, double, double>,
                        GMRESData<double, float, double>,
                        Vector<double>>(
    SparseMatrix<double>&, SparseMatrix<double, float>&,
    GMRESData<double, double, double>&, GMRESData<double, float, double>&,
    Vector<double>&, Vector<double>&, bool, bool, TestGMRESData&);
#else
template int TestGMRES< SparseMatrix<double>,
                        SparseMatrix<float>,
                        GMRESData<double, double, double>,
                        GMRESData<float, float, float>,
                        Vector<double>>(
    SparseMatrix<double>&, SparseMatrix<float>&,
    GMRESData<double, double, double>&, GMRESData<float, float, float>&,
    Vector<double>&, Vector<double>&, bool, bool, TestGMRESData&);
#endif
