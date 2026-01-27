
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
 @file ComputeGEMMT_ref.cpp

 HPGMP routine for computing GEMM transpose (dot-products)
 */
#if !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP) & !defined(HPGMP_WITH_BLAS)

#ifndef HPGMP_NO_MPI
#include "Utils_MPI.hpp"
#endif

#include "ComputeGEMMT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"

template<class MultiVector_type, class SerialDenseMatrix_type>
int ComputeGEMMT_ref(const local_int_t m, const local_int_t n, const local_int_t k,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type& A, const MultiVector_type& B,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type& C)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    typedef typename MultiVector_type::scalar_type scalarA_type;
    typedef typename SerialDenseMatrix_type::scalar_type scalarC_type;

    const scalarA_type one(1.0);
    const scalarA_type zero(0.0);

    // Input serial dense vector
    const scalarA_type* const Av = A.values();
    const scalarA_type* const Bv = B.values();
    scalarC_type* const Cv       = C.values();

    // GEMM on HOST CPU
    double t0;
    TICK();
    if (beta == zero) {
        for (local_int_t i = 0; i < m * n; i++) Cv[i] = zero;
    } else if (beta != one) {
        for (local_int_t i = 0; i < m * n; i++) Cv[i] *= beta;
    }

    if (alpha == one) {
        for (local_int_t i = 0; i < m; i++) {
            for (local_int_t j = 0; j < n; j++) {
                for (local_int_t h = 0; h < k; h++) {
                    Cv[i + j * m] += Av[h + i * k] * Bv[j + j * k];
                }
            }
        }
    } else {
        for (local_int_t i = 0; i < m; i++) {
            for (local_int_t j = 0; j < n; j++) {
                for (local_int_t h = 0; h < k; h++) {
                    Cv[i + j * m] += alpha * Av[h + i * k] * Bv[j + j * k];
                }
            }
        }
    }
    TIME(C.time1);

#ifndef HPGMP_NO_MPI
    // Use MPI's reduce function to collect all partial sums
    TICK();
    int size; // Number of MPI processes, My process ID
    MPI_Comm_size(A.get_comm(), &size);
    if (size > 1) {
        MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarC_type>::getType();
        MPI_Allreduce(MPI_IN_PLACE, Cv, m * n, MPI_SCALAR_TYPE, MPI_SUM, A.get_comm());
    }
    TIME(C.time2);
#else
    C.time2 = 0.0;
#endif

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template int ComputeGEMMT_ref< MultiVector<double>, SerialDenseMatrix<double> >(
    int, int, int, double, MultiVector<double> const&, MultiVector<double> const&, double, SerialDenseMatrix<double>&);

template int ComputeGEMMT_ref< MultiVector<float>, SerialDenseMatrix<float> >(
    int, int, int, float, MultiVector<float> const&, MultiVector<float> const&, float, SerialDenseMatrix<float>&);

#endif
