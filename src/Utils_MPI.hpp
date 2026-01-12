//This file should contain MPI utils only!!

#ifndef HPGMP_UTILS_HPP
#define HPGMP_UTILS_HPP

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#include "DataTypes.hpp"

// MpiTypeTraits (from Teuchos)
template<class T>
class MpiTypeTraits
{
public:
    static MPI_Datatype getType()
    {
        return MPI_DATATYPE_NULL;
    }
    static MPI_Op getSumOp()
    {
        return MPI_NO_OP;
    }
};

//! Specialization for T = double (from Teuchos)
template<>
class MpiTypeTraits<double>
{
public:
    //! MPI_Datatype corresponding to the type T.
    static MPI_Datatype getType()
    {
        return MPI_DOUBLE;
    }
    static MPI_Op getSumOp()
    {
        return MPI_SUM;
    }
};

//! Specialization for T = float (from Teuchos).
template<>
class MpiTypeTraits<float>
{
public:
    //! MPI_Datatype corresponding to the type T.
    static MPI_Datatype getType()
    {
        return MPI_FLOAT;
    }
    static MPI_Op getSumOp()
    {
        return MPI_SUM;
    }
};

#endif // ifndef HPGMP_NO_MPI
#endif
