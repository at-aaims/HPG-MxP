
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
 @file ComputeGEMVT_gpu.cpp

 HPGMP routine for computing GEMV transpose (dot-products)
 */
#if defined(HPGMP_WITH_BLAS)

#ifndef HPGMP_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "cblas.h"
#include "ComputeGEMVT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"


template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT_ref(const local_int_t m, const local_int_t n,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & y) {

  HPGMP_RANGE_PUSH(__FUNCTION__);

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  assert(x.localLength >= m); // Test vector lengths
  assert(y.m >= n);
  assert(y.n == 1);


  // Input serial dense vector 
  const scalarX_type * const xv = x.values;
  const scalarA_type * const Av = A.values;

  // Output serial dense vector
  scalarY_type * const yv = y.values;

  // Perform GEMV on host
  double t0; TICK();
  const int ione = 1;
  if (std::is_same<scalarX_type, double>::value) {
    cblas_dgemv(CblasColMajor, CblasTrans, m, n,
                alpha, (double*)Av, m,
                       (double*)xv, ione,
                beta,  (double*)yv, ione);
  } else if (std::is_same<scalarX_type, float>::value) {
    cblas_sgemv(CblasColMajor, CblasTrans, m, n,
                alpha, (float*)Av, m,
                       (float*)xv, ione,
                beta,  (float*)yv, ione);
  }
  TIME(y.time1);

#ifndef HPGMP_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  TICK();
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
  MPI_Allreduce(MPI_IN_PLACE, yv, n, MPI_SCALAR_TYPE, MPI_SUM, A.comm);
  TIME(y.time2);
#else
  y.time2 = 0.0;
#endif

  HPGMP_RANGE_POP(__FUNCTION__);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMVT_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, Vector<double> const&, double, SerialDenseMatrix<double> &);

template
int ComputeGEMVT_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, Vector<float> const&, float, SerialDenseMatrix<float> &);

#endif
