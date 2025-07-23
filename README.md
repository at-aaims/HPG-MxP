# High Performance Generalized minimal residual - Mixed-Precision (HPG-MxP) Benchmark

The [original implementation](https://github.com/hpg-mxp/hpg-mxp) is written by
Ichitaro Yamazaki, Jennifer Loe, Christian Glusa, Sivasankaran Rajamanickam,
Piotr Luszczek, and Jack Dongarra.
Please refer to that repository for documentation on the original implementation.

This version is maintained by the National Center for Computational Sciences
at Oak Ridge National Laboratory.

## Introduction

HPG-MxP is a software package that performs a fixed number of multigrid preconditioned
(using a Gauss-Seidel smoother) Generalized minimal residual (PGMRES) iterations in order to
solve a possibly nonsymmetric large sparse linear system of equations.
It is designed to be a benchmark to measure a computer's performance for sparse linear algebra
workloads typical in scientific computing while allowing the use of mixed precision methods.
The solution is required to have convergence characteristics and accuracy similar to
double precision GMRES.
It is based on the High Performance Conjugate Gradient Benchmark ([HPCG](https://github.com/hpcg-benchmark/hpcg))
which restricts all implementations to use only the IEEE double precision format (FP64).
This is somewhat analogous to how the High Performance Linpack
([HPL](https://icl.utk.edu/hpl/index.html)) benchmark solves a large dense linear system
and restricts the numerical format to FP64, while the [HPL-MxP](https://github.com/ROCm/rocHPL-MxP)
benchmark solves an easier system but uses iterative refinement to allow the use of
lower precision formats in the factorization stage while still obtaining a solution as accurate
as using only FP64.

The HPG-MxP rating is is a weighted GFLOP/s (billion floating operations per second) value
that is composed of the operations performed in the PGMRES iteration phase over
the time taken.  The overhead time of problem construction and any modifications to improve
performance are amortized over several iterations (the amortization weight) and added to the runtime.

Integer arrays have global and local
scope (global indices are unique across the entire distributed memory system,
local indices are unique within a memory image).  Integer data for global/local
indices have three modes:

* 32/32 - global and local integers are 32-bit
* 64/32 - global integers are 64-bit, local are 32-bit
* 64/64 - global and local are 64-bit.

These various modes are required in order to address sufficiently big problems
if the range of indexing goes above 2^31 (roughly 2.1B), or to conserve storage
costs if the range of indexing is less than 2^31.

This HPG-MxP software package requires the availibility on your system of an
implementation of the Message Passing Interface (MPI) if enabling the MPI
build of HPG-MxP, a compiler that supports OpenMP syntax, and an accelerator compiler
and library ecosystem.

This implementation uses several ideas from AMD's HPCG implementation [rocHPCG](https://github.com/ROCm/rocHPCG).

## Installation

See the [installation](doc/install.md) instructions.
Please note that only CMake builds are currently supported.

## Valid Runs

Please see the instructions to [run](doc/run.md) the code.

HPG-MxP can be run in just a few minutes from start to finish.  However, official
runs must be at least 1800 seconds (30 minutes) as reported in the output file.
The Quick Path option is an exception for machines that are in production mode
prior to broad availability of an optimized version of HPGMP for a given platform.
In this situation (which should be confirmed by sending a note to the HPGMP Benchmark
owners) the Quick Path option can be invoked by setting the run time parameter equal
to 0 (zero).

A valid run must also execute a problem size that is large enough so that data
arrays accessed in the CG iteration loop do not fit in the cache of the device
in a way that would be unrealistic in a real application setting.  Presently this
restriction means that the problem size should be large enough to occupy a
significant fraction of *main memory*, at least 1/4 of the total.

Future memory system architectures may require restatement of the specific memory
size requirements.  But the guiding principle will always be that the problem
size should reflect what would be reasonable for a real sparse iterative solver.

## Bugs

Known problems and bugs with this release are documented in the file
`BUGS`.

## Citing

The additional optimizations, features and other changes over the reference implementation are
described in the paper:
Kashi et al., "Scaling the memory wall using mixed-precision - HPG-MxP on an exascale machine",
arXiv:2507.11512 [cs.DC], July 2025.
