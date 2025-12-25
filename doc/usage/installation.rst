.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. **NOTE**: this file is referenced by README.md:
.. if changing the former, update the latter!!

.. highlight:: cmake

.. _infrastructure:

Installation
============

Celeritas is designed to be easy to install for a multitude of use cases.

Package managers
----------------

Celeritas is available through the Spack_ package manager, which is designed
for HPC environments and scientific software. The `Celeritas Spack package`_
supports a wide range of configuration options including GPU acceleration
(CUDA and HIP), geometry backends (VecGeom and ORANGE), I/O implementations
(ROOT, HepMC3), and profiling (Perfetto).
This makes it easy to install Celeritas with the exact feature set
needed for the user's application.
The typical HEP use case, which requires Geant4 and VecGeom, is built by default.

.. _Spack: https://spack.io
.. _Celeritas Spack package: https://packages.spack.io/package.html?name=celeritas

.. _dependencies:

Dependencies
------------

Celeritas is built using modern CMake_. It has multiple dependencies to operate
as a full-featured code, but each dependency can be individually disabled as
needed.

.. _CMake: https://cmake.org

The code requires external dependencies to build with full functionality, but
none of them need to be installed externally for the code to work.
Most can be omitted entirely to enable limited development on experimental HPC
systems or personal machines with fewer available components. Items with an
asterisk ``*`` in the category below will be fetched from the
internet if required but not available on the user's system.

.. tabularcolumns:: lll

.. csv-table::
   :header: Component, Category, Description
   :widths: 10, 10, 20

   CLI11_, Runtime*, "Command line parsing"
   CUDA_, Runtime, "GPU computation"
   Geant4_, Runtime, "Physics data and user integration"
   G4EMLOW_, Runtime, "EM physics model data"
   G4VG_, Runtime, "Geant4-to-VecGeom translation"
   HepMC3_, Runtime, "Event input"
   HIP_, Runtime, "GPU computation"
   LArSoft_, Runtime, "LAr detector framework integration"
   libpng_, Runtime, "PNG output for raytracing"
   nljson_, Runtime*, "Simple text-based I/O for diagnostics and program setup"
   "`Open MPI`_", Runtime, "Shared-memory parallelism"
   ROOT_, Runtime, "Input and output"
   VecGeom_, Runtime, "On-device navigation of GDML-defined detector geometry"
   Breathe_, Docs, "Generating code documentation inside user docs"
   Doxygen_, Docs, "Code documentation"
   Sphinx_, Docs, "User documentation"
   sphinxbib_, Docs, "Reference generation for user documentation"
   clang-format_, Development, "C++ code formatting"
   codespell_, Development, "Spell checking"
   CMake_, Development, "Build system"
   Git_, Development, "Repository management"
   pre-commit_, Development, "Formatting enforcement"
   GoogleTest_, Development*, "Test harness"
   Perfetto_, Development*, "CPU profiling"

.. _breathe: https://github.com/michaeljones/breathe#readme
.. _clang-format: https://clang.llvm.org/docs/ClangFormat.html
.. _codespell: https://github.com/codespell-project/codespell
.. _CLI11: https://cliutils.github.io/CLI11/book/
.. _CMake: https://cmake.org
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _Doxygen: https://www.doxygen.nl
.. _G4VG: https://github.com/celeritas-project/g4vg
.. _G4EMLOW: https://geant4.web.cern.ch/support/download
.. _Geant4: https://geant4.web.cern.ch/support/download
.. _Git: https://git-scm.com
.. _GoogleTest: https://github.com/google/googletest
.. _HepMC3: http://hepmc.web.cern.ch/hepmc/
.. _HIP: https://docs.amd.com
.. _LArSoft: https://larsoft.github.io/LArSoftWiki/
.. _libpng: http://www.libpng.org/
.. _nljson: https://github.com/nlohmann/json
.. _Open MPI: https://www.open-mpi.org
.. _Perfetto: https://perfetto.dev/
.. _pre-commit: https://pre-commit.com
.. _ROOT: https://root.cern
.. _Sphinx: https://www.sphinx-doc.org/
.. _sphinxbib: https://pypi.org/project/sphinxcontrib-bibtex/
.. _VecGeom: https://gitlab.cern.ch/VecGeom/VecGeom


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


.. _configuration:

Configuration options
^^^^^^^^^^^^^^^^^^^^^

The interactive ``ccmake`` tool is highly recommended for exploring the
Celeritas configuration options, since it provides both documentation *and* an
easy way to toggle through all the valid options.

``CELERITAS_USE_{package}``
  Enable features of the given dependency. The configuration will fail if the
  dependent package is not found.

``CELERITAS_BUILTIN_{package}``
  Force a package to be built from an internally downloaded copy (when true/on)
  or externally installed code (when false/off).

``CELERITAS_BUILD_{DOCS|TESTS}``
  Build optional documentation and/or tests.

``CELERITAS_CORE_GEO``
  Select the geometry package used by the Celeritas stepping loop. Valid
  options include VecGeom, Geant4, and ORANGE. There are limits on
  compatibility: Geant4 is not compatible with GPU-enabled or OpenMP builds,
  and VecGeom is not compatible with HIP.

``CELERITAS_CORE_RNG``
  Select the pseudorandom number generator. Current options are
  platform-dependent implementations of XORWOW.

``CELERITAS_DEBUG``
  Enable detailed runtime assertions. These *will* slow down the code
  considerably. A separate ``CELERITAS_DEBUG_DEVICE`` option allows debug
  checking inside device code to be enabled/disabled since the generated code
  substantially increases kernel size and build time in addition to affecting
  performance more substantially.

``CELERITAS_OPENMP``
  Choose between no multithreaded OpenMP parallelism (``disabled``),
  ``event``-level parallelism for the ``celer-sim`` app, and ``track``-level
  parallelism. OpenMP *should* be disabled with multithreaded Geant4 but *will*
  work correctly with single-threaded applications.

``CELERITAS_REAL_TYPE``
  Choose between ``double`` and ``float`` real numbers across the codebase.
  This is currently experimental.

``CELERITAS_UNITS``
  Choose the native Celeritas unit system: see :ref:`the unit
  documentation <api_units>`.

``CELERITAS_CODATA``
  Choose the default set of experimentally measured CODATA constants:
  see :ref:`the constants documentation <api_constants>`.

Celeritas libraries (generally) use CMake-provided default properties. These
can be changed with standard `CMake variables`_ such as ``BUILD_SHARED_LIBS`` to
enable shared libraries, ``CMAKE_POSITION_INDEPENDENT_CODE``, etc.

.. _CMake variables: https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html

Toolchain installation
----------------------

The recommended way to install dependencies is with ``Spack``,
an HPC-oriented package manager that includes numerous scientific packages,
including those used in HEP. Celeritas includes a Spack development environment
at :file:`scripts/spack.yaml` that describes the code's full suite
of dependencies (including testing and documentation). To install these
dependencies:

- Clone and load Spack following its `getting started instructions
  <https://spack.readthedocs.io/en/latest/getting_started.html>`_.
- If using CUDA: run ``spack external find cuda`` to inform Spack of the
  existing installation.
- Create the Celeritas development environment with ``spack env create
  celeritas scripts/spack.yaml``.
- Tell Spack to default to building with CUDA support with
  the command ``spack -e celeritas config add packages:all:variants:"cxxstd=17 +cuda
  cuda_arch=<ARCH>"``, where ``<ARCH>`` is the numeric portion of the
  `CUDA architecture flags
  <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.
- Install all the dependencies with ``spack -e celeritas install``.

The current Spack environment for full-featured development is:

.. literalinclude:: ../../scripts/spack.yaml
   :language: yaml

With this environment (with CUDA enabled), all Celeritas tests should be
enabled and all should pass. Celeritas is build-compatible with older versions
of some dependencies (e.g., Geant4@10.6 and VecGeom@1.2.7), but some tests may
fail, indicating a change in behavior or a bug fix in that package.
Specifically, older versions of VecGeom have shapes and configurations that are
incompatible on GPU with new CMS detector descriptions.

Building Celeritas
------------------

Once the Celeritas Spack environment has been installed, set your shell's environment
variables (``PATH``, ``CMAKE_PREFIX_PATH``, ...) by activating it.

To clone the latest development version of Celeritas:

.. code-block:: console

   $ git clone https://github.com/celeritas-project/celeritas.git

or download it from the GitHub-generated `zip file`_.

Celeritas includes a :ref:`build_script` that helps setting up the code on development machines.
It includes environment files for quickly getting started on systems including NERSC's Perlmutter_, ORNL's ExCL_ and Frontier_ systems, and ANL's JLSE_.

.. code-block:: console

   $ cd celeritas
   $ ./scripts/build.sh base

You can, of course, manually build as a

.. code-block:: console

   $ cd celeritas
   $ spack env activate celeritas # if using spack
   $ mkdir build && cd build
   $ cmake ..
   $ make && ctest

.. _zip file: https://github.com/celeritas-project/celeritas/archive/refs/heads/develop.zip
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/
.. _ExCL: https://docs.olcf.ornl.gov/systems/excl_user_guide.html
.. _Frontier: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
.. _JLSE: https://www.jlse.anl.gov/

.. _cmake_presets:

CMake Presets
-------------

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
build, and test.

.. _build_script:

Build script
------------

The :file:`scripts/build.sh` helper script automates common development tasks
for building Celeritas. It intelligently configures the build environment by:

- detecting the system hostname and loading the corresponding environment file
  from :file:`scripts/env/{hostname}.sh` (if available)
- linking the appropriate CMake user presets from
  :file:`scripts/cmake-presets/{hostname}.json`
- detecting and enabling ccache_ for faster rebuilds
- configuring, building, and testing, and
- installing pre-commit hooks for code quality checks.

The script accepts a preset name as the first argument, followed by any
additional CMake configuration arguments. For example:

.. code-block:: console

   $ ./scripts/build.sh default -DCELERITAS_DEBUG=ON

Common presets include ``minimal`` (fewest dependencies), ``default``
(detecting dependencies from the environment), and ``full`` (all optional
features enabled).
System-specific presets such as ``reldeb-orange`` may also be available.

The provided environment files for certain shared HPC systems may change key
variables such as ``XDG_CACHE_HOME``.
In such cases, the build script will modify the shell's rc file to source the
environment script at login.

.. _ccache: https://ccache.dev/
