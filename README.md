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
setting the `CELERITAS_BUILD_DOCS=ON` configuration option and running
`ninja doc` (user) or `ninja doxygen` (developer).

[inline-docs]: doc/index.rst
[user-docs]: https://celeritas-project.github.io/celeritas/user/index.html
[dev-docs]: https://celeritas-project.github.io/celeritas/dev/index.html

# Installation for applications

The easiest way to install Celeritas as a library/app is with Spack:
- Follow these steps to install [Spack][spack-start].
```console
# Install Spack
git clone --depth=2 https://github.com/spack/spack.git
# Add Spack to the shell environment
# For bash/zsh/sh (See [spack-start] for other shell)
. spack/share/spack/setup-env.sh
```
- Install Celeritas with
```console
spack install celeritas
```
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

[spack-start]: https://spack.readthedocs.io/en/latest/getting_started.html

# Integrating into a Geant4 app

In the simplest case, integration requires a few small changes to your user
applications, with many more details described in
[integration overview][integration].

You first need to find Celeritas in your project's CMake file, and change
library calls to support VecGeom's use of CUDA RDC:
```diff
+find_package(Celeritas 0.6 REQUIRED)
 find_package(Geant4 REQUIRED)
@@ -36,3 +37,4 @@ else()
   add_executable(trackingmanager-offload trackingmanager-offload.cc)
-  target_link_libraries(trackingmanager-offload
+  celeritas_target_link_libraries(trackingmanager-offload
+    Celeritas::accel
     ${Geant4_LIBRARIES}
```

One catch-all include exposes the Celeritas high-level offload classes to user
code:
```diff
--- example/geant4/trackingmanager-offload.cc
+++ example/geant4/trackingmanager-offload.cc
@@ -31,2 +31,4 @@

+// Celeritas
+#include <CeleritasG4.hh>

```

Celeritas uses the run action to set up and tear down cleanly:
```diff
--- example/accel/trackingmanager-offload.cc
+++ example/accel/trackingmanager-offload.cc
@@ -133,2 +138,3 @@ class RunAction final : public G4UserRunAction
     {
+        TMI::Instance().BeginOfRunAction(run);
     }
@@ -136,2 +142,3 @@ class RunAction final : public G4UserRunAction
     {
+        TMI::Instance().EndOfRunAction(run);
     }
```

And integrates into the tracking loop primarily using the `G4TrackingManager`
interface:
```diff
--- example/accel/trackingmanager-offload.cc
+++ example/accel/trackingmanager-offload.cc
@@ -203,4 +235,8 @@ int main()

+    auto& tmi = TMI::Instance();
+
     // Use FTFP_BERT, but use Celeritas tracking for e-/e+/g
     auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
+    physics_list->RegisterPhysics(
+        new celeritas::TrackingManagerConstructor(&tmi));
```

More flexible alternatives to this high level interface, compatible with other
run manager implementations and older versions of Geant4, are described in the
manual.

[integration]: https://celeritas-project.github.io/celeritas/user/usage/integration.html

# Installation for developers

Since Celeritas is still under very active development, you may be installing it
for development purposes. The [installation documentation][installation] has a
complete description of the code's dependencies and installation process for
development.

As an example, if you have the [Spack][spack] package manager
installed and want to do development on a CUDA system with Ampere-class graphics
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
# Set up
# Configure, build, and test with a default development configure
$ ./scripts/build.sh dev
```

If you don't use Spack but have all the dependencies you want (Geant4,
GoogleTest, VecGeom, etc.) in your `CMAKE_PREFIX_PATH`, you can configure and
build Celeritas as you would any other project:
```console
$ mkdir build && cd build
$ cmake ..
$ make && ctest
```

> [!NOTE]
> It is **highly** recommended to use the `build.sh` script to set up your
> environment, even when not using Spack. The first time you run it, edit the
> `CMakeUserPresets.json` symlink it creates, and submit it in your next pull
> request.

Celeritas guarantees full compatibility and correctness only on the
combinations of compilers and dependencies tested under continuous integration.
See the configure output from the [GitHub runners][runners] for the full list of combinations.
- Compilers
    - GCC 11, 12, 14
    - Clang 10, 15, 18
    - MSVC 19
    - GCC 11.5 + NVCC 12.6
    - ROCm Clang 18
- Platforms
    - Linux x86_64, ARM
    - Windows x86_64
- C++ standard
    - C++17 and C++20
- Dependencies:
    - Geant4 11.0.4
    - VecGeom 1.2.10

Partial compatibility and correctness is available for an extended range of
Geant4:
- 10.5-10.7: no support for tracking manager offload
- 11.0: no support for fast simulation offload

Note also that navigation bugs in Geant4 and VecGeom older than the versions
listed above *will* cause failures in some geometry-related unit tests. Future
behavior changes in external packages may also cause failures.

Since we compile with extra warning flags and avoid non-portable code, most
other compilers *should* work.
The full set of configurations is viewable on CI platform [GitHub Actions][gha].
Compatibility fixes that do not cause newer versions to fail are welcome.

[installation]: https://celeritas-project.github.io/celeritas/user/usage/installation.html
[spack]: https://github.com/spack/spack
[runners]: https://github.com/celeritas-project/celeritas/actions/workflows/push.yml
[gha]: https://github.com/celeritas-project/celeritas/actions

# Development

<!-- This section should be kept in sync with the doc/development files -->

See the [contribution guide][contributing-guidelines] for the contribution process,
[the development guidelines][development-guidelines] for further
details on coding in Celeritas, and [the administration guidelines][administration-guidelines] for community standards and roles.

[contributing-guidelines]: https://celeritas-project.github.io/celeritas/user/development/contributing.html
[development-guidelines]: https://celeritas-project.github.io/celeritas/user/development/coding.html
[administration-guidelines]: https://celeritas-project.github.io/celeritas/user/development/administration.html

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

and its Zenodo release metadata for version 0.6: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15281110.svg)](https://doi.org/10.5281/zenodo.15281110)

> Seth R. Johnson, Amanda Lund, Julien Esseiva, Philippe Canal, Elliott Biondo, Hayden Hollenbeck, Stefano Tognini, Lance Bullerwell, Soon Yung Jun, Guilherme Lima, Damien L-G, & Sakib Rahman. (2025). Celeritas 0.6 (0.6.0). Github. https://doi.org/10.5281/zenodo.15281110

A continually evolving list of works authored by (or with content authored by)
core team members is continually updated at [our publications page](https://github.com/celeritas-project/celeritas/blob/doc/gh-pages-base/publications.md)
and displayed on [the official project web site](https://celeritas.ornl.gov/).
