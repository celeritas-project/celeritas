# Celeritas

The Celeritas project implements HEP detector physics on GPU accelerator
hardware with the ultimate goal of supporting the massive computational
requirements of the [HL-LHC upgrade][HLLHC].

[HLLHC]: https://home.cern/science/accelerators/high-luminosity-lhc

# Documentation

Most of the Celeritas documentation is readable through the codebase through a
combination of [static RST documentation][inline-docs] and Doxygen-markup
comments in the source code itself. The full [Celeritas user
documentation][user-docs] (including selected code documentation incorporated
by Breathe) and the [Celeritas code documentation][dev-docs] are mirrored on
our GitHub pages site. You can generate these yourself (if the necessary
prerequisites are installed) by
setting the `CELERITAS_BUILD_DOCS=ON` configuration option and running `ninja
doc` (user) or `ninja doxygen` (developer).

[inline-docs]: doc/index.rst
[user-docs]: https://celeritas-project.github.io/celeritas/user/index.html
[dev-docs]: https://celeritas-project.github.io/celeritas/dev/index.html

# Installation for applications

The easiest way to install Celeritas as a library/app is with Spack:
- Follow these steps to install [Spack][spack-start].
```console
# Install Spack
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
# Add Spack to the shell environment
# For bash/zsh/sh (See [spack-start] for other shell)
. spack/share/spack/setup-env.sh
```
- Install Celeritas with
```console
spack install celeritas
```console
- Add the Celeritas installation to your `PATH` with:
```console
spack load celeritas
```

To install a GPU-enabled Celeritas build, you might have to make sure that VecGeom is also built with CUDA
support if installing `celeritas+vecgeom`, which is the default geometry.
To do so, set Spack up its CUDA usage:
```console
. spack/share/spack/setup-env.sh
# Set up CUDA
$ spack external find cuda
# Optionally set the default configuration. Replace "cuda_arch=80"
# with your target architecture
$ spack config add packages:all:variants:"cxxstd=17 +cuda cuda_arch=80"
```
and install Celeritas with this configuration:
```console
$ spack install celeritas
```
If Celeritas was installed with a different configuration do
```console
$ spack install --fresh celeritas
```
If you need to set a default configuration
```console
$ spack install celeritas +cuda cuda_arch=80
```


Then see the "Downstream usage as a library" section of the [installation
documentation][install] for how to use Celeritas in your application or framework.

[spack-start]: https://spack.readthedocs.io/en/latest/getting_started.html
[install]: https://celeritas-project.github.io/celeritas/user/introduction/installation.html

# Installation for developers

Since Celeritas is still under heavy development and is not yet full-featured
for downstream integration, you are likely installing it for development
purposes. The [installation documentation][install] has a
complete description of the code's dependencies and installation process for
development.

As an example, if you have the [Spack][spack] package manager
installed and want to do development on a CUDA system with Volta-class graphics
cards, execute the following steps from within the cloned Celeritas source
directory:
```console
# Set up CUDA (optional)
$ spack external find cuda
# Install celeritas dependencies
$ spack env create celeritas scripts/spack.yaml
$ spack env activate celeritas
$ spack config add packages:all:variants:"cxxstd=17 +cuda cuda_arch=80"
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
See the configure output from the [GitHub runners](https://github.com/celeritas-project/celeritas/actions/workflows/push.yml) for the full list of combinations.
- Compilers and standard:
    - GCC 8, 11, 12, 14
    - Clang 10, 15, 18
    - GCC 11.5 + NVCC 12.6
    - ROCm Clang 18
    - C++17 and C++20
- Dependencies:
    - Geant4 11.0.4
    - VecGeom 1.2.10

Partial compatibility and correctness is available for an extended range of
Geant4:
- 10.5-10.7: no support for tracking manager offload
- 11.0: no support for fast simulation offload
- 11.1-11.3: [no support for default Rayleigh scattering cross section](see
  https://github.com/celeritas-project/celeritas/issues/1091)

Note also that navigation bugs in older versions of Geant4 and VecGeom can
cause test failures in Celeritas.

Since we compile with extra warning flags and avoid non-portable code, most
other compilers *should* work.
The full set of configurations is viewable on CI platform [GitHub Actions][gha]).
Compatibility fixes that do not cause newer versions to fail are welcome.

[spack]: https://github.com/spack/spack
[install]: https://celeritas-project.github.io/celeritas/user/introduction/installation.html
[gha]: https://github.com/celeritas-project/celeritas/actions

# Development

See the [contribution guide][contributing-guidelines] for the contribution process,
[the development guidelines][development-guidelines] for further
details on coding in Celeritas, and [the administration guidelines][administration-guidelines] for community standards and roles.

[contributing-guidelines]: https://celeritas-project.github.io/celeritas/user/development/contributing.html
[development-guidelines]: https://celeritas-project.github.io/celeritas/user/development/coding.html
[administration-guidelines]: https://celeritas-project.github.io/celeritas/user/appendix/administration.html

# Directory structure

| **Directory** | **Description**                                       |
|---------------|-------------------------------------------------------|
| **app**       | Source code for installed executable applications     |
| **cmake**     | Implementation code for CMake build configuration     |
| **doc**       | Code documentation and manual                         |
| **example**   | Example applications and input files                  |
| **external**  | Automatically fetched external CMake dependencies     |
| **scripts**   | Development and continuous integration helper scripts |
| **src**       | Library source code                                   |
| **test**      | Unit tests                                            |

# Citing Celeritas

<!-- This section should be kept in sync with the CITATIONS.cff file -->

If using Celeritas in your work, we ask that you cite the following article:

> Johnson, Seth R., Amanda Lund, Philippe Canal, Stefano C. Tognini, Julien Esseiva, Soon Yung Jun, Guilherme Lima, et al. 2024. “Celeritas: Accelerating Geant4 with GPUs.” EPJ Web of Conferences 295:11005. https://doi.org/10.1051/epjconf/202429511005.

See also its [DOECode](https://www.osti.gov/doecode/biblio/94866) registration:

> Johnson, Seth R., Amanda Lund, Soon Yung Jun, Stefano Tognini, Guilherme Lima, Philippe Canal, Ben Morgan, Tom Evans, and Julien Esseiva. 2022. “Celeritas.” https://doi.org/10.11578/dc.20221011.1.

A continually evolving list of works authored by (or with content authored by)
core team members is continually updated at [our publications page](https://github.com/celeritas-project/celeritas/blob/doc/gh-pages-base/publications.md)
and displayed on [the official project web site](https://celeritas.ornl.gov/).
