#ifndef HPGMP_ELL_MULTICOLOR_GS_HPP
#define HPGMP_ELL_MULTICOLOR_GS_HPP

#include "ell_matrix.hpp"
#include "Vector.hpp"

template <typename mscalar, typename vscalar>
int ell_multicolor_gs(const ELLMatrix<mscalar> *A, const Vector<vscalar> *r,
                     Vector<vscalar> *x);

template <typename mscalar, typename vscalar>
int ell_multicolor_gs_zero_initial(const ELLMatrix<mscalar> *A,
        const Vector<vscalar> *r, Vector<vscalar> *x);

#endif
