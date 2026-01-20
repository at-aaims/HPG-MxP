
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
 @file ComputeGEMV_blas.cpp

 HPGMP routine for computing GEMV (vector-update)
 */
#if defined(HPGMP_WITH_BLAS)

#include "cblas.h"
#include "ComputeGEMV_ref.hpp"
#include "hpgmp.hpp"
#include "Profiling.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMV_ref(const local_int_t m, const local_int_t n,
                    const typename MultiVector_type::scalar_type alpha, const MultiVector_type& A, const SerialDenseMatrix_type& x,
                    const typename Vector_type::scalar_type beta, const Vector_type& y)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    typedef typename MultiVector_type::scalar_type scalarA_type;
    typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
    typedef typename Vector_type::scalar_type scalarY_type;

    const int ione = 1;
    const scalarA_type one(1.0);
    const scalarA_type zero(0.0);

    assert(x.m >= n); // Test vector lengths
    assert(x.n == 1);
    assert(y.localLength >= m);

    // Input serial dense vector
    const scalarX_type* const xv = x.values;

    scalarA_type* const Av = A.values;
    scalarY_type* const yv = y.values;

    if ((std::is_same<scalarX_type, double>::value && std::is_same<scalarY_type, double>::value && std::is_same<scalarA_type, double>::value) ||
        (std::is_same<scalarX_type, float >::value && std::is_same<scalarY_type, float >::value && std::is_same<scalarA_type, float >::value)) {

        // Perform GEMV on host
        if (std::is_same<scalarX_type, double>::value) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, n,
                        alpha, (double*)Av, m,
                        (double*)xv, ione,
                        beta, (double*)yv, ione);
        } else if (std::is_same<scalarX_type, float>::value) {
            cblas_sgemv(CblasColMajor, CblasNoTrans, m, n,
                        alpha, (float*)Av, m,
                        (float*)xv, ione,
                        beta, (float*)yv, ione);
        }
    } else
    {
        //HPGMP_vout << " Mixed-precision GEMV not supported" << std::endl;
        // GEMV on HOST CPU
        if (beta == zero) {
            for (local_int_t i = 0; i < m; i++) yv[i] = zero;
        } else if (beta != one) {
            for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
        }

        if (alpha == one) {
            for (local_int_t j = 0; j < n; j++)
                for (local_int_t i = 0; i < m; i++) {
                    yv[i] += Av[i + j * m] * xv[j];
                }
        } else {
            for (local_int_t j = 0; j < n; j++)
                for (local_int_t i = 0; i < m; i++) {
                    yv[i] += alpha * Av[i + j * m] * xv[j];
                }
        }
    }

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template int ComputeGEMV_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >(
    int, int, double, MultiVector<double> const&, SerialDenseMatrix<double> const&, double, Vector<double> const&);

template int ComputeGEMV_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >(
    int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, float, Vector<float> const&);


// mixed
template int ComputeGEMV_ref< MultiVector<float>, Vector<double>, SerialDenseMatrix<float> >(
    int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&);

#endif
