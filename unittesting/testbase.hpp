#ifndef HPGMP_UNITTESTING_TESTBASE_HPP
#define HPGMP_UNITTESTING_TESTBASE_HPP

#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<typename scalar>
struct TestFixture
{
    const Geometry* geom;
    SparseMatrix<scalar>* A;
    Vector<scalar>*x, b, xexact;
};

int run_all_tests(comm_type comm, const HPGMP_Params& params);

#endif
