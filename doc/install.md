# HPG-MxP installation

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

* `HPGMP_ENABLE_LONG_LONG` option enables a build with `long long`
  type used for global indices (`ON` by default).

* `HPGMP_ENABLE_OPENMP` option enables a build with OPENMP enabled
  (`OFF` by default).

* Compile with modest debugging turned on: `-DHPGMP_DEBUG`

* Compile with sparse matrix arrays allocated contiguously. This option
  may be helpful on systems with pre-fetch: `-DHPGMP_CONTIGUOUS_ARRAYS`

* Compile with voluminous debugging information turned on: `-DHPGMP_DETAILED_DEBUG`

* Compile with MPI disabled: `-DHPGMP_NO_MPI`

* Compile without OpenMP enabled: `-DHPGMP_NO_OPENMP`

* Enable detail timers: `-DHPGMP_DETAILED_TIMING`

* Use reference implementations (as opposed to optimized versions): `HPGMP_BUILD_REFERENCE`

By default HPGMP will::

* Turn on MPI support.
* Turn on OpenMP support.
* not display detailed timing information.

There are more options available. Please see the ``CMakeLists.txt`` file
and the `option()` commands therein.

These options may be changed with `ccmake` command or given directly
to the `cmake` invocation with the `-D` prefix.

## Build

In the build directory, issue `cmake --build .`

### Example Build

An example CMake line for an AMD MI200 system is (assuming you are already in the `BUILD_DIR`,
and that a MPI installation is available in the environment):

```
cmake -DHPGMP_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=installs/gmp-base path/to/hpgmp_source_dir
make -j8
make install
```
