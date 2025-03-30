
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

#ifndef MYTIMER_HPP
#define MYTIMER_HPP

#include "device_ctx.hpp"

double mytimer(void);

void fence();

void fence(stream_t stream);

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  fence(); t0 = mytimer()      //!< record current time in 't0'
#define TOCK(t) fence(); t += mytimer() - t0 //!< store time difference in 't' using time in 't0'
#define TIME(t) fence(); t  = mytimer() - t0 //!< store time difference in 't' using time in 't0'

#define START_T()  fence(); start_t = mytimer()      //!< record current time in 'start_t'
#define STOP_T(t)  fence(); t += mytimer() - start_t //!< store time difference in 't' using time in 'start_t'

// Stream-specific timers
//! record current time in 't0' after synchronizing the given device stream.
#define TICKS(stream)  fence(stream); t0 = mytimer()
//! store time difference in 't' using time in 't0' after syncing the given stream.
#define TOCKS(stream, t) fence(stream); t += mytimer() - t0

#endif // MYTIMER_HPP
