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
 @file CheckAspectRatio.cpp

 HPGMP routine
 */

#include <algorithm>

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#endif

#include "hpgmp.hpp"

#include "CheckAspectRatio.hpp"

int CheckAspectRatio(double smallest_ratio, int x, int y, int z, const char* what, bool DoIo)
{
    double current_ratio = std::min(std::min(x, y), z) / double(std::max(std::max(x, y), z));

    if (current_ratio < smallest_ratio) { // ratio of the smallest to the largest
        if (DoIo) {
            HPGMP_fout << "The " << what << " sizes (" << x << "," << y << "," << z //
                       << ") are invalid because the ratio min(x,y,z)/max(x,y,z)=" << current_ratio //
                       << " is too small (at least " << smallest_ratio << " is required)." << std::endl;
            HPGMP_fout << "The shape should resemble a 3D cube. Please adjust and try again." << std::endl;
            HPGMP_fout.flush();
        }

#ifndef HPGMP_NO_MPI
        MPI_Abort(MPI_COMM_WORLD, 127);
#endif

        return 127;
    }

    return 0;
}
