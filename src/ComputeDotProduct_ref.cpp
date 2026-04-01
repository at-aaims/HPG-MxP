
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
 @file ComputeDotProduct_ref.cpp

 HPGMP routine
 */
#if !defined(HPGMP_WITH_BLAS) & !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)

#include "mytimer.hpp"
#ifndef HPGMP_NO_MPI
#include <mpi.h>
#include "Utils_MPI.hpp"
#endif
#include "ComputeDotProduct_ref.hpp"
#include "hpgmp.hpp"

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
template<class Vector_type, class scalar_type>
int ComputeDotProduct_ref(const local_int_t n, const Vector_type& x, const Vector_type& y,
                          scalar_type& result, double& time_allreduce)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    assert(x.local_length() >= n); // Test vector lengths
    assert(y.local_length() >= n);

    scalar_type local_result(0.0);

    const scalar_type* xv = x.values();
    const scalar_type* yv = y.values();
    if (yv == xv) {
#ifndef HPGMP_NO_OPENMP
        // clang-format off
        #pragma omp parallel for reduction(+ : local_result)
        // clang-format on
#endif
        for (local_int_t i = 0; i < n; i++) local_result += xv[i] * xv[i];
    } else {
#ifndef HPGMP_NO_OPENMP
        // clang-format off
        #pragma omp parallel for reduction(+ : local_result)
        // clang-format on
#endif
        for (local_int_t i = 0; i < n; i++) local_result += xv[i] * yv[i];
    }

#ifndef HPGMP_NO_MPI
    // Use MPI's reduce function to collect all partial sums
    double t0 = mytimer();
    int size; // Number of MPI processes, My process ID
    MPI_Comm_size(x.get_comm(), &size);

    result = local_result;
    if (size > 1) {
        MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType();
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_SCALAR_TYPE, MPI_SUM, x.get_comm());
    }
    time_allreduce += mytimer() - t0;
#else
    time_allreduce += 0.0;
#endif

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeDotProduct_ref<Vector<double> >(
    int, Vector<double> const&, Vector<double> const&, double&, double&);

template int ComputeDotProduct_ref<Vector<float> >(
    int, Vector<float> const&, Vector<float> const&, float&, double&);

#endif
