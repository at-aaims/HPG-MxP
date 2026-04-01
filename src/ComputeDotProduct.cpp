
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
 @file ComputeDotProduct.cpp

 HPGMP routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
template<class Vector_type, class scalar_type>
int ComputeDotProduct(const local_int_t n, const Vector_type& x, const Vector_type& y,
                      scalar_type& result, double& time_allreduce, bool& isOptimized)
{

    // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
    isOptimized = false;
    return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeDotProduct< Vector<double> >(
    int, Vector<double> const&, Vector<double> const&, double&, double&, bool&);

template int ComputeDotProduct< Vector<float> >(
    int, Vector<float> const&, Vector<float> const&, float&, double&, bool&);
