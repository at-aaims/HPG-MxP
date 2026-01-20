
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
 @file ComputeWAXPBY_ref.cpp

 HPGMP routine
 */
#if !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)

#include "ComputeWAXPBY_ref.hpp"
#include "hpgmp.hpp"
#include <cassert>

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "Profiling.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY_ref(const local_int_t n,
                      const typename VectorX_type::scalar_type alpha,
                      const VectorX_type& x,
                      const typename VectorY_type::scalar_type beta,
                      const VectorY_type& y,
                      VectorW_type& w)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    assert(x.localLength >= n); // Test vector lengths
    assert(y.localLength >= n);

    // quick return
    if (n <= 0) {
        HPGMP_RANGE_POP(__FUNCTION__);
        return 0;
    }

    typedef typename VectorX_type::scalar_type scalarX_type;
    typedef typename VectorY_type::scalar_type scalarY_type;
    typedef typename VectorW_type::scalar_type scalarW_type;

    scalarX_type* const xv = x.values;
    scalarY_type* const yv = y.values;
    scalarW_type* const wv = w.values;

    if (alpha == 1.0) {
#ifndef HPGMP_NO_OPENMP
        // clang-format off
        #pragma omp parallel for
        // clang-format on
#endif
        for (local_int_t i = 0; i < n; i++) wv[i] = xv[i] + beta * yv[i];
    } else if (beta == 1.0) {
#ifndef HPGMP_NO_OPENMP
        // clang-format off
        #pragma omp parallel for
        // clang-format on
#endif
        for (local_int_t i = 0; i < n; i++) wv[i] = alpha * xv[i] + yv[i];
    } else {
#ifndef HPGMP_NO_OPENMP
        // clang-format off
        #pragma omp parallel for
        // clang-format on
#endif
        for (local_int_t i = 0; i < n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
    }

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template int ComputeWAXPBY_ref< Vector<double>, Vector<double>, Vector<double> >(
    int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&);

template int ComputeWAXPBY_ref< Vector<float>, Vector<float>, Vector<float> >(
    int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&);


// mixed
template int ComputeWAXPBY_ref< Vector<double>, Vector<float>, Vector<double> >(
    int, double, Vector<double> const&, float, Vector<float> const&, Vector<double>&);

#endif
