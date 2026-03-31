
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
 @file ComputeResidual.cpp

 HPGMP routine
 */
#ifndef HPGMP_NO_MPI
#include <mpi.h>
#include "Utils_MPI.hpp"
#endif
#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "Vector.hpp"

#ifdef HPGMP_DETAILED_DEBUG
#include <fstream>
#include "hpgmp.hpp"
#endif

#include <cmath> // needed for fabs
#include "ComputeResidual.hpp"
#ifdef HPGMP_DETAILED_DEBUG
#include <iostream>
#endif

/*!
  Routine to compute the inf-norm difference between two vectors where:

  @param[in]  n        number of vector elements (local to this processor)
  @param[in]  v1, v2   input vectors
  @param[out] residual pointer to scalar_type value; on exit, will contain result: inf-norm difference

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class Vector_type>
int ComputeResidual(const local_int_t n, const Vector_type& v1, const Vector_type& v2, typename Vector_type::scalar_type& residual)
{

    typedef typename Vector_type::scalar_type scalar_type;
    const scalar_type* v1v = v1.values();
    const scalar_type* v2v = v2.values();
    scalar_type local_residual(0.0);

#ifndef HPGMP_NO_OPENMP
    // clang-format off
    #pragma omp parallel shared(local_residual, v1v, v2v)
    // clang-format on
    {
        scalar_type threadlocal_residual(0.0);
        // clang-format off
        #pragma omp for
        // clang-format on
        for (local_int_t i = 0; i < n; i++) {
            scalar_type diff = std::fabs(v1v[i] - v2v[i]);
            if (diff > threadlocal_residual) threadlocal_residual = diff;
        }
        // clang-format off
        #pragma omp critical
        // clang-format on
        {
            if (threadlocal_residual > local_residual) local_residual = threadlocal_residual;
        }
    }
#else // No threading
    for (local_int_t i = 0; i < n; i++) {
        scalar_type diff = std::fabs(v1v[i] - v2v[i]);
        if (diff > local_residual) local_residual = diff;
#ifdef HPGMP_DETAILED_DEBUG
        HPGMP_fout << " Computed, exact, diff = " << v1v[i] << " " << v2v[i] << " " << diff << std::endl;
#endif
    }
#endif

#ifndef HPGMP_NO_MPI
    // Use MPI's reduce function to collect all partial sums
    scalar_type global_residual  = 0;
    MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType();
    MPI_Allreduce(&local_residual, &global_residual, 1, MPI_SCALAR_TYPE, MPI_MAX, v1.get_comm());
    residual = global_residual;
#else
    residual = local_residual;
#endif

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeResidual< Vector<double> >(
    int, Vector<double> const&, Vector<double> const&, double&);

template int ComputeResidual< Vector<float> >(
    int, Vector<float> const&, Vector<float> const&, float&);
