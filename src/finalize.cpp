
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

#include <fstream>

#include "hpgmp.hpp"
#include "DataTypes.hpp"

/*!
  Closes the I/O stream used for logging information throughout the HPGMP run.

  @return returns 0 upon success and non-zero otherwise

  @see HPGMP_Init
*/
int HPGMP_Finalize(void)
{
    HPGMP_fout.close();
    return 0;
}
