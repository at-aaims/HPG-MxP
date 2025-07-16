# High Performance Generalized minimal residual - Mixed-Precision (HPG-MxP)

## Configuration

This version of HPG-MxP provides only CMake-based configuration.

### Configuration based on CMake

For configuration based on CMake, a build directory, `BUILD_DIR`, is first created.

The compiler and linker are specified with the CMake-specific flags.
Additionally, HPGMP-specific options of interest include:

* `HPGMP_ENABLE_DEBUG` option enables a build with debugging output
  (`OFF` by default).

* `HPGMP_ENABLE_DETAILED_DEBUG` option enables a build with detailed
  debugging output (`OFF` by default).

* `HPGMP_ENABLE_MPI` option enables a build with MPI enabled (`OFF` by default).

* `HPGMP_ENABLE_LONG_LONG` option enables a build with ``long long``
  type used for global indices (`ON` by default).

* `HPGMP_ENABLE_OPENMP` option enables a build with OPENMP enabled
  (`OFF` by default).

* Compile with modest debugging turned on: `-DHPGMP_DEBUG`

* Compile with sparse matrix arrays allocated contiguously. This option
  may be helpful on systems with pre-fetch: `-DHPGMP_CONTIGUOUS_ARRAYS`

* Compile with voluminous debugging information turned on: `-DHPGMP_DETAILED_DEBUG`

* Compile with MPI disabled:

    -DHPGMP_NO_MPI

* Compile without OpenMP enabled:

    -DHPGMP_NO_OPENMP

* Enable detail timers:

    -DHPGMP_DETAILED_TIMING


By default HPGMP will::

* Turn on MPI support.
* Turn on OpenMP support.
* not display detailed timing information.

There are more options available. Please see the ``CMakeLists.txt`` file
and the `option()` commands therein.

These options may be changed with `ccmake` command or given directly
to the `cmake` invocation with the `-D` prefix.

An example CMake line for an AMD MI200 system is (assuming you are already in the `BUILD_DIR`):

```
cmake -DHPGMP_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=installs/gmp-base path/to/hpgmp_source_dir
```

## Build

In the build directory, issue
cmake --build .

### Example Build

As an example, let's use a Linux cluster and create a file called
``Make.Linux`` in the ``setup`` directory right under the top-level
directory.

For in-source build, we type ``make setup/Make.Linux`` which creates the
executable file called ``bin/xhpgmp``.

For out-of-source build, we create the build directory called
`build_Linux` and go to that directory::

    mkdir build_Linux
    cd build_Linux

Then, while in the ``build_Linux`` directory we type::

    /path/to/hpgmp/configure Linux
    make

This creates the executable file ``bin/xhpgmp``.

## Test

For a quick check, go to the ``bin`` directory and run the HPGMP
executable as follows::

    mpirun -np 8 xhpgmp

Note that this will use the default ``hpgmp.dat`` file. If you'd like to
change the size of local dimensions of the problem to ``NX=32``,
``NY=24``, ``NZ=16`` then run the following::

    mpirun -np 8 xhpgmp 32 24 16

You can also specific size and runtime parameters using ``--nx``,
``--ny``, ``--nz``, ``--rt``.  For example, for specifying the local
grid dimensions to be NX=NY=NZ=16, and the timed phase execution limit
of 30 minutes (1800 seconds) you can use

    mpirun -np 4 xhpgmp --nx=16 --rt=1800

## Tuning

Most of the performance  parameters can be tuned by modifying the input
file `hpgmp.dat`. See the file `TUNING` in the top-level directory.

