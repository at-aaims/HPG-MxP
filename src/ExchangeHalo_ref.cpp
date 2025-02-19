
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
 @file ExchangeHalo.cpp

 HPGMP routine
 */

// Compile this routine only if running with MPI
#if !defined(HPGMP_NO_MPI) & !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)
#include <mpi.h>
#include "Utils_MPI.hpp"
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include "mytimer.hpp"
#include <cstdlib>

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
template<class SparseMatrix_type, class Vector_type>
void ExchangeHalo_ref(const SparseMatrix_type & A, Vector_type & x) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();

  // Extract Matrix pieces
  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  scalar_type * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  scalar_type * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(A.comm, &size);
  MPI_Comm_rank(A.comm, &rank);
  if (size == 1) return;

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  scalar_type * x_external = (scalar_type *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  double t0 = 0.0;
  TICK();
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm, request+i);
    x_external += n_recv;
  }
  double time2 = 0.0;
  TOCK(time2);


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  TICK();
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];
  double time1 = 0.0;
  TOCK(time1);

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  TICK();
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm);
    sendBuffer += n_send;
  }

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }
  TOCK(time2);

  x.time1 = time1; x.time2 = time2;
  delete [] request;

  return;
}


/* --------------- *
 * specializations *
 * --------------- */

template
void ExchangeHalo_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double>&);

template
void ExchangeHalo_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float>&);

#endif // ifndef HPGMP_NO_MPI
