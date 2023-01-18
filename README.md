# Celeritas

The Celeritas project implements HEP detector physics on GPU accelerator
hardware with the ultimate goal of supporting the massive computational
requirements of the [HL-LHC upgrade][HLLHC].

[HLLHC]: https://home.cern/science/accelerators/high-luminosity-lhc

# Documentation

Most of the Celeritas documentation is readable through the codebase through a
combination of [static RST documentation](doc/index.rst) and Doxygen-markup
comments in the source code itself. The full [Celeritas user
documentation][user-docs] (including selected code documentation incorporated
by Breathe) and the [Celeritas code documentation][dev-docs] are mirrored on
our GitHub pages site. You can generate these yourself by
setting the `CELERITAS_BUILD_DOCS=ON` configuration option and running `ninja
doc` (user) or `ninja doxygen` (developer). A continuously updated version of
the [static Celeritas user documentation][rtd] (without API documentation) is
hosted on `readthedocs.io`.

[user-docs]: https://celeritas-project.github.io/celeritas/user/index.html)
[dev-docs]: https://celeritas-project.github.io/celeritas/dev/index.html)
[rtd]: https://celeritas.readthedocs.io/en/latest/

# Installation for developers

Since Celeritas is still under heavy development and is not yet full-featured
for downstream integration, you are likely installing it for development
purposes. The [infrastructure documentation][infra] has a
complete description of the code's dependencies and installation process for
development.

As an example, if you have the [Spack][spack] package manager
installed and want to do development on a CUDA system with Volta-class graphics
cards, execute the following steps:
```console
# Set up CUDA (optional)
$ spack external find cuda
$ spack config add packages:all:variants:"+cuda cuda_arch=70"
# Install celeritas dependencies
$ spack env create celeritas scripts/spack.yaml
$ spack env activate celeritas
$ spack install
# Configure, build, and test
$ ./build.sh base
```

If you don't use Spack but have all the dependencies you want (Geant4,
googletest, VecGeom, etc.) in your `CMAKE_PREFIX_PATH`, you can configure and
build Celeritas as you would any other project:
```console
$ mkdir build && cd build
$ cmake ..
$ make && ctest
```

Celeritas guarantees full compatibility and correctness only on the
combinations of compilers and dependencies tested under continuous integration.
Currently supported compilers are GCC 11.2 + NVCC 11.8, and HIP-Clang 15.0, but
since we compile with extra warning flags and avoid non-portable code, most
other compilers *should* work.
Currently Geant4 11.0 and VecGeom 1.2 are the only versions that are guaranteed
to work, but older versions might be OK.
The full set of configurations is viewable on [the CI web site][jenkins].
Compatibility fixes that do not cause newer versions to fail are welcome.

[spack]: https://github.com/spack/spack
[infra]: doc/infrastructure.rst
[jenkins]: https://cloud.cees.ornl.gov/jenkins-ci/blue/organizations/jenkins/Celeritas/activity?branch=master

# Installation for applications

The easiest way to install Celeritas as a library/app is with Spack:
- Follow the first two steps above to install [Spack][spack-start] and set up its CUDA usage.
- Install Celeritas with `spack install celeritas`
- Use `spack load celeritas` to add the installation to your `PATH`.

Then see the "Downstream usage as a library" section of the [infrastructure
documentation][infra] for how to use Celeritas in your application or framework.

[spack-start]: https://spack.readthedocs.io/en/latest/getting_started.html
[infra]: doc/infrastructure.rst

# Development

See the [contribution guide](CONTRIBUTING.rst) for the contribution process,
[the development guidelines](doc/appendices/development.rst) for further
details on coding in Celeritas, and [the administration guidelines](doc/appendices/administration.rst) for community standards and roles.

# Citing Celeritas

If using Celeritas in your work, we ask that you cite the code using its
[DOECode](https://www.osti.gov/doecode/biblio/94866) registration:

> Johnson, Seth R., Amanda Lund, Soon Yung Jun, Stefano Tognini, Guilherme Lima, Paul Romano, Philippe Canal, Ben Morgan, and Tom Evans. “Celeritas,” July 2022. https://doi.org/10.11578/dc.20221011.1.

Additional references for code implementation details, benchmark problem
results, etc., can be found in our continually evolving [citation
file](doc/_static/celeritas.bib).
