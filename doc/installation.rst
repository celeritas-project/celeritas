.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: cmake

.. _infrastructure:

************
Installation
************


Dependencies
============

Celeritas is built using modern CMake_. It has multiple dependencies to operate
as a full-featured code, but each dependency can be individually disabled as
needed.

.. _CMake: https://cmake.org

The code requires external dependencies to build with full functionality.
However, any combination of these requirements can be omitted to enable
limited development on experimental HPC systems or personal machines with
fewer available components.

.. tabularcolumns:: lll

.. csv-table::
   :header: Component, Category, Description
   :widths: 10, 10, 20

   CUDA_, Runtime, "GPU computation"
   Geant4_, Runtime, "Preprocessing physics data for a problem input"
   G4EMLOW_, Runtime, "EM physics model data"
   HepMC3_, Runtime, "Event input"
   HIP_, Runtime, "GPU computation"
   nljson_, Runtime, "Simple text-based I/O for diagnostics and program setup"
   "`Open MPI`_", Runtime, "Shared-memory parallelism"
   ROOT_, Runtime, "Input and output"
   SWIG_, Runtime, "Low-level Python wrappers"
   VecGeom_, Runtime, "On-device navigation of GDML-defined detector geometry"
   Breathe_, Docs, "Generating code documentation inside user docs"
   Doxygen_, Docs, "Code documentation"
   Sphinx_, Docs, "User documentation"
   sphinxbib_, Docs, "Reference generation for user documentation"
   clang-format_, Development, "Code formatting enforcement"
   CMake_, Development, "Build system"
   Git_, Development, "Repository management"
   GoogleTest_, Development, "Test harness"

.. _CMake: https://cmake.org
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _Doxygen: https://www.doxygen.nl
.. _G4EMLOW: https://geant4.web.cern.ch/support/download
.. _Geant4: https://geant4.web.cern.ch/support/download
.. _Git: https://git-scm.com
.. _GoogleTest: https://github.com/google/googletest
.. _HepMC3: http://hepmc.web.cern.ch/hepmc/
.. _HIP: https://docs.amd.com
.. _Open MPI: https://www.open-mpi.org
.. _ROOT: https://root.cern
.. _SWIG: http://swig.org
.. _Sphinx: https://www.sphinx-doc.org/
.. _VecGeom: https://gitlab.cern.ch/VecGeom/VecGeom
.. _breathe: https://github.com/michaeljones/breathe#readme
.. _clang-format: https://clang.llvm.org/docs/ClangFormat.html
.. _nljson: https://github.com/nlohmann/json
.. _sphinxbib: https://pypi.org/project/sphinxcontrib-bibtex/


Ideally you will build Celeritas with all dependencies to gain the full
functionality of the code, but there are circumstances in which you may not
have (or want) all the dependencies or features available. By default, the CMake code in
Celeritas queries available packages and sets several
``CELERITAS_USE_{package}``
options based on what it finds, so you have a good chance of successfully
configuring Celeritas on the first go. Some optional features
will error out in the configure if their required
dependencies are missing, but they will update the CMake cache variable so that
the next configure will succeed (with that component disabled).

Toolchain installation
======================

The recommended way to install dependencies is with ``Spack``,
an HPC-oriented package manager that includes numerous scientific packages,
including those used in HEP. Celeritas includes a Spack environment at
:file:`scripts/spack.yaml` that describes the code's full suite
of dependencies (including testing and documentation). To install these
dependencies:

- Clone and load Spack following its `getting started instructions
  <spack-start>`.
- If using CUDA: run ``spack external find cuda`` to inform Spack of the existing
  installation, and tell Spack to default to building with CUDA support with
  the command ``spack config add packages:all:variants:"+cuda
  cuda_arch=<ARCH>"``, where ``<ARCH>`` is the numeric portion of the `CUDA
  architecture flags <cuda-arch>`.
- Create the Celeritas development environment with ``spack env create
  celeritas scripts/spack.yaml``.
- Install all the dependencies with ``spack -e celeritas install``.

.. _Spack: https://github.com/spack/spack
.. _spack-start: https://spack.readthedocs.io/en/latest/getting_started.html
.. _cuda-arch: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

Building Celeritas
==================

Once the Celeritas Spack environment has been installed, set your shell's environment
variables (``PATH``, ``CMAKE_PREFIX_PATH``, ...) by activating it. Then, you
can configure, build, and test using the provided helper script:

.. code-block:: console

   $ spack env activate celeritas
   $ ./scripts/build.sh base

or manually with:

.. code-block:: console

   $ spack env activate celeritas
   $ mkdir build && cd build
   $ cmake ..
   $ make && ctest

CMake Presets
=============

To manage multiple builds with different
configure options (debug or release, VecGeom or ORANGE), you can use the
CMake presets provided by Celeritas via the ``CMakePresets.json`` file for CMake
3.21 and higher:

.. code-block:: console

   $ cmake --preset=default

The three main options are "minimal", "default", and "full", which all set
different expectations for available dependencies.

.. note::

   If your CMake version is too old, you may get an unhelpful message:

   .. code-block:: console

      CMake Error: Could not read presets from celeritas: Unrecognized "version"
      field

   which is just a poor way of saying the version in the ``CMakePresets.json``
   file is newer than that version knows how to handle.

If you want to add your own set of custom options and flags, create a
``CMakeUserPresets.json`` file or, if you wish to contribute on a regular
basis, create a preset at :file:`scripts/cmake-presets/{HOSTNAME}.json` and
call ``scripts/build.sh {preset}`` to create the symlink, configure the preset,
build, and test. See :file:`scripts/README.md` in the code repository for more
details.

Downstream usage as a library
=============================

The Celeritas library is most easily used when your downstream app is built with
CMake. It should require a single line to initialize::

   find_package(Celeritas REQUIRED CONFIG)

and if VecGeom or CUDA are disabled a single line to link::

   target_link_libraries(mycode PUBLIC Celeritas::celeritas)

Because of complexities involving CUDA Relocatable Device Code, linking when
using both CUDA and VecGeom requires an additional include and the replacement
of ``target_link_libraries`` with a customized version::

  include(CeleritasLibrary)
  celeritas_target_link_libraries(mycode PUBLIC Celeritas::celeritas)

The test project at :file:`scripts/ci/test-installation` demonstrates how to
use Celeritas as a library with a short standalone CMake project.
