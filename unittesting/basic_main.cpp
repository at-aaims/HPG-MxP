#include "hpgmp.hpp"

#include "testbase.hpp"

/*!
  Test driver program: Construct geometry, run tests, teardown.

  @return Returns zero on success and a non-zero value otherwise.
 */
int main(int argc, char * argv[])
{
#ifndef HPGMP_NO_MPI
    MPI_Init(&argc, &argv);
#endif
    HPGMP_Init(&argc, &argv);
#ifndef HPGMP_NO_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    comm_type comm = 0;
#endif
    HPGMP_Params params;
    HPGMP_Init_Params(&argc, &argv, params, comm);
    params.numThreads = 1;

    int ierr = run_all_tests(comm, params);

    HPGMP_Finalize();
#ifndef HPGMP_NO_MPI
    MPI_Finalize();
#endif
    return ierr;
}
