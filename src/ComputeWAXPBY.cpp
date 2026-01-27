
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
 @file ComputeWAXPBY.cpp

 HPGMP routine
 */

#include "ComputeWAXPBY.hpp"
#ifdef HPGMP_REFERENCE
#include "ComputeWAXPBY_ref.hpp"
#else
#include "ComputeWAXPBY_opt.hpp"
#endif

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY(const local_int_t n,
                  const typename VectorX_type::scalar_type alpha,
                  const VectorX_type& x,
                  const typename VectorY_type::scalar_type beta,
                  const VectorY_type& y,
                  VectorW_type& w,
                  bool& isOptimized)
{

    // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
#ifdef HPGMP_REFERENCE
    isOptimized = false;
    return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
#else
    isOptimized = true;
    return ComputeWAXPBY_opt(n, alpha, x, beta, y, w);
#endif
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template int ComputeWAXPBY< Vector<double>, Vector<double>, Vector<double> >(
    int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&, bool&);

template int ComputeWAXPBY< Vector<float>, Vector<float>, Vector<float> >(
    int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&, bool&);


// mixed
template int ComputeWAXPBY< Vector<double>, Vector<float>, Vector<double> >(
    int, double, Vector<double> const&, float, Vector<float> const&, Vector<double>&, bool&);
