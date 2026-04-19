.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. **NOTE**: this file is referenced by README.md:
.. if changing the former, update the latter!!

.. highlight:: console

.. _infrastructure:

Installation
============

Celeritas is designed to be easy to install for a multitude of use cases.
Users should probably use Spack or another package manager to install
Celeritas, and developers should install the software dependencies, configure,
and build themselves.

If *not* using a package manager, you should download the latest `develop archive`_ or `release version`_, or clone with Git:

.. code::

   $ git clone https://github.com/celeritas-project/celeritas.git

Then, with the necessary dependencies installed, you can configure and build.

.. code::

   $ cd celeritas
   $ mkdir build && cd build
   $ cmake ..
   $ ninja

The following section describes details of configuration and more advanced use
cases for building.

.. _develop archive: https://github.com/celeritas-project/celeritas/archive/refs/heads/develop.zip
.. _release version: https://github.com/celeritas-project/celeritas/releases

.. _install_spack:

Installing with Spack
---------------------

Celeritas is available through the Spack_ package manager, which is designed
for HPC environments and scientific software, including HEP packages.
The `Celeritas Spack package`_
supports a wide range of configuration options including GPU acceleration
(CUDA and HIP), geometry backends (VecGeom and ORANGE), I/O implementations
(ROOT, HepMC3), and profiling (Perfetto).
This makes it easy to install Celeritas with the exact feature set
needed for the user's application.
The typical HEP use case, which requires Geant4 and VecGeom, is built by default.
.. code::

   # Use an Nvidia A100 GPU with VecGeom
   $ spack install celeritas +cuda cuda_arch=80 +vecgeom
   # Use an AMD MI250x
   $ spack install celeritas +rocm amdgpu_target=gfx90a
   # Add celeritas to your $PATH
   $ spack load celeritas

.. _Spack: https://spack.io
.. _Celeritas Spack package: https://packages.spack.io/package.html?name=celeritas

.. _dependencies:

Software dependencies
---------------------

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
   DD4hep_, Runtime, "HEP detector framework integration"
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
.. _DD4hep: https://dd4hep.web.cern.ch
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
have (or want) all the dependencies or features available.

.. note::

   The ``LArSoft`` metapackage in Celeritas looks for a few specific components
   of the full LArSoft stack: cetmodules_, art_, and LArSoft's data object
   model.

.. _cetmodules: https://fnalssi.github.io/cetmodules/
.. _art: https://art.fnal.gov/

.. _spack_deps:

Installing dependencies with Spack
----------------------------------

When building locally for development or deployment,
the recommended way to install dependencies is with Spack_.
Celeritas includes Spack development environments
at :file:`scripts/spack/env-{which}.yaml` for development and execution.
To install these dependencies for basic use with an Nvidia GPU:

- Clone and load Spack following its `getting started`_ instructions.
- Run ``spack external find cuda`` to inform Spack of the
  existing CUDA installation.
- Create the Celeritas development environment with
  ``spack env create celeritas scripts/spack/env-cuda.yaml``
- Activate the environment with ``spack env activate celeritas``
- Tell Spack to default to building with CUDA support with
  the command ``spack config add "packages:all:prefer:cuda_arch=<ARCH>"``,
  where ``<ARCH>`` is the numeric portion of the CUDA `architecture flags`_.
- Install all the dependencies with ``spack install``.

The dependency requirements for Celeritas are:

.. literalinclude:: ../../scripts/spack/packages.yaml
   :language: yaml

and the full list of packages used by Celeritas is:

.. literalinclude:: ../../scripts/spack/env-full.yaml
   :language: yaml
   :start-at: specs:
   :end-before: view:

With this environment (with CUDA enabled), all Celeritas tests should be
enabled and all should pass.
Celeritas is build-compatible with older versions of some dependencies (e.g.,
Geant4@10.6 and VecGeom@1.2.7), but some tests may fail, indicating a change in
behavior or a bug fix in that package.

Once the Celeritas Spack environment has been installed, set your shell's
environment variables (``PATH``, ``CMAKE_PREFIX_PATH``, ...) by activating it
with ``spack env activate celeritas``.

.. _getting started: https://spack.readthedocs.io/en/latest/getting_started.html
.. _architecture flags: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

.. _build_script:

Developer build script
----------------------

Celeritas includes a build script, :file:`scripts/build.sh`, that configures
and builds the code on development machines.
It includes environment files for quickly getting started on systems including NERSC's Perlmutter_, ORNL's ExCL_ and Frontier_ systems, and ANL's JLSE_.

It intelligently configures the build environment by:

- detecting the system hostname and loading the corresponding environment file
  from :file:`scripts/env/{hostname}.sh` (if available)
- detecting and loading apptainer-specific setups using environment variables
- linking the appropriate CMake user presets from
  :file:`scripts/cmake-presets/{system}.json`
- detecting and enabling ccache_ for faster rebuilds
- configuring, building, and testing, and
- installing pre-commit hooks for code quality checks.

The script accepts a preset name as the first argument, followed by any
additional CMake configuration arguments. For example:

.. code::

   $ ./scripts/build.sh default -DCELERITAS_DEBUG=ON

Common presets include ``minimal`` (fewest dependencies), ``default``
(detecting dependencies from the environment), and ``full`` (all optional
features enabled).
System-specific presets such as ``reldeb-orange`` may also be available.

The provided environment files for certain shared HPC systems may change key
variables such as ``XDG_CACHE_HOME``.
In such cases, the build script will modify the shell's rc file to source the
environment script at login.

.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/
.. _ExCL: https://docs.excl.ornl.gov
.. _Frontier: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
.. _JLSE: https://www.jlse.anl.gov/
.. _ccache: https://ccache.dev/

.. _configuration:

Configuring Celeritas
---------------------

By default, the CMake code in Celeritas queries available packages and sets
several :samp:`CELERITAS_USE_{package}` options based on what it finds, so you
have a good chance of successfully configuring Celeritas on the first go.
Some optional features will fail during configuration if their required
dependencies are missing, but they will update the CMake cache variable so that
the next configure will succeed (with that component disabled).

The interactive ``ccmake`` tool is highly recommended for exploring the
Celeritas configuration options, since it provides both documentation *and* an
easy way to toggle through all the valid options.

Additionally, CMake presets are included for both general and machine-specific
cases.
These presets bundle sets of useful options and compiler flags.

CMake variables
^^^^^^^^^^^^^^^

:samp:`CELERITAS_USE_{package}`
  Enable features of the given dependency. The configuration will fail if the
  dependent package is not found.

:samp:`CELERITAS_BUILTIN_{package}`
  Force a package to be built from an internally downloaded copy (when true/on)
  or externally installed code (when false/off).

:samp:`CELERITAS_BUILD_{DOCS|TESTS}`
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

``CELERITAS_RESEED``
  Choose when the random number generator is reseeded.  Valid options include
  trackslot and track.  With trackslot, each trackslot gets a unique random
  number generator.  With track, every particle track gets a unique random
  number generator.  The track option provides improved reproducibility at
  greater computational expense.

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

.. _cmake_presets:

CMake presets
^^^^^^^^^^^^^

To manage multiple builds with different
configure options (debug or release, VecGeom or ORANGE), you can use the
CMake presets provided by Celeritas via the ``CMakePresets.json`` file for CMake
3.21 and higher:

.. code::

   $ cmake --preset=default

The three main options are "minimal", "default", and "full", which all set
different expectations for available dependencies.

.. note::

   If your CMake version is too old, you may get an unhelpful message:

   .. code::

      CMake Error: Could not read presets from celeritas: Unrecognized "version"
      field

   which is just a poor way of saying the version in the ``CMakePresets.json``
   file is newer than that version knows how to handle.

If you want to add your own set of custom options and flags, create a
``CMakeUserPresets.json`` file or, if you wish to contribute on a regular
basis, create a preset at :file:`scripts/cmake-presets/{HOSTNAME}.json` and
call ``scripts/build.sh {preset}`` to create the symlink, configure the preset,
build, and test.

.. _build_ups:

UPS for LArSoft
---------------

Since LArSoft and DUNE (see :ref:`plugins_larsoft`) require many infrastructure
components specific to the Fermilab UPS_ packaging system and art_ framework,
it is difficult to install on a typical user system.
However, the necessary dependencies are available as "build products" via the
DUNE/LArSoft/Fermilab CVMFS_ distribution and built through a standard
Fermilab-provided Apptainer_ image.
Building Celeritas for LArSoft is trivial once the LArSoft development
environment has been set up.

.. _MRB: https://cdcvs.fnal.gov/redmine/projects/mrb/wiki/MrbUserGuide
.. _CVMFS: https://cvmfs.readthedocs.io/en/stable/
.. _Apptainer: https://apptainer.org/docs/user/main/quick_start.html
.. _UPS: https://cdcvs.fnal.gov/redmine/projects/ups/wiki/Getting_Started_Using_UPS

.. note:: UPS and these images are in the process of being replaced with a
   Spack toolchain. If you are using a Spack-based distribution of
   larsoft/dunesw already, you should be able to install Celeritas with the
   standard instructions above.

.. _apptainer_env:

Apptainer
^^^^^^^^^

UPS-based builds always happen within a containerized system. These
instructions demonstrate container execution for two use cases: using CUDA on the ExCL milan2_ system, and without CUDA on Fermilab's ``scisoftbuild01`` machine.

.. _milan2: https://docs.excl.ornl.gov/system-overview/milan

To enable CUDA, launch the ``fnal-dev-sl7:latest`` Apptainer_ image, stored on
CVMFS_, with CUDA forwarding enabled (and the CUDA directory forwarded via
``-B``):

.. literalinclude:: ../../scripts/env/excl.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_DOC_APPTAINER
   :end-before: END_DOC_APPTAINER

This command is wrapped into the ``apptainer-fnal`` shell command when
:file:`scripts/env/excl.sh` is sourced.

The ``--ipc --pid`` options ask Apptainer to give the container isolated
interprocess communication and process ID namespaces for VM-like process
isolation.

On Fermilab machines, most of which require Kerberos authentication and do
*not* have CUDA support, omit the ``--nv`` flag and forward the hosts files.

.. literalinclude:: ../../scripts/env/scisoftbuild01.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_DOC_APPTAINER
   :end-before: END_DOC_APPTAINER

This script forwards:

- the cvmfs and ``/opt`` directories to provide build tools and products,
- the higher-performance temporary build directories in ``$SCRATCHDIR``,
- the home directory for source code and shell scripts,
- ``$XDG_RUNTIME_DIR`` to allow ssh-agent forwarding, and
- network configuration files.

.. important:: Because the ``fnal-dev-sl7`` uses a *very* old operating system,
   the default LArG4 installation will likely fail to load when enabling CUDA
   with the ``--nv`` flag, which forwards a number of host libraries to the
   container. If this happens, you will see an error:

   .. code:: none

     Unable to load requested library .../liblarg4_Services_LArG4Detector_service.so
     /lib64/libc.so.6: version 'GLIBC_2.38' not found (required by /.singularity.d/libs/libGLX.so.0)

   This is due to Geant4's visualization functionality (which uses OpenGL).
   It can be fixed by commenting out the lines in
   :file:`{/etc}/apptainer/nvliblist.conf` that start with libGL and libgl.

.. tip:: Apptainer overrides the ``$PS1`` shell prompt variable even if your
   forwarded home directory overrides it. To override it inside the apptainer,
   define the ``APPTAINERENV_PS1`` environment variable in the bare-metal
   machine (i.e., the login node). For example:

   .. code:: sh

      APPTAINERENV_PS1='\D{%b %d %H:%M:%S} \u@\h|$APPTAINER_NAME:\w\n$ '


.. _ups_mrb:

UPS and MRB
^^^^^^^^^^^

To set up Celeritas dependencies for minimal LArSoft development:

.. literalinclude:: ../../scripts/env/fnal-dev-sl7.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_DOC_UPS
   :end-before: END_DOC_UPS

The ``-q`` qualifiers_ denote the compiler version and flags.
These dependencies are loaded automatically when using the ``build.sh`` script
inside the Apptainer image.

.. _qualifiers: https://cdcvs.fnal.gov/redmine/projects/cet-is-public/wiki/AboutQualifiers

.. tip::

   Use the command :samp:`ups list -aK+ {package}` to list available packages.

Alternatively, for integration into DUNESW_ development environment:

.. code::

   $ source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
   Setting up larsoft UPS area... /cvmfs/larsoft.opensciencegrid.org
   Setting up DUNE UPS area... /cvmfs/dune.opensciencegrid.org/products/dune/
   $ setup dunesw v10_20_00d00 -q e26:prof

If using MRB_ with at least one repository (i.e. you called ``mrb g ...``),
``cmake`` will be available in your ``$PATH``.

.. _DUNESW: https://github.com/DUNE/dunesw/releases
.. _MRB: https://cdcvs.fnal.gov/redmine/projects/mrb/wiki/MrbUserGuide

Installing Celeritas
^^^^^^^^^^^^^^^^^^^^

Celeritas does not currently have a UPS package.
Instead, build and install it like any other CMake package, using the build
script, the LArSoft preset (which activates ``CELERITAS_USE_LArSoft`` and disables unnecessary :ref:`dependencies`), or manually:

.. code::

   $ git clone https://github.com/celeritas-project/celeritas.git
   $ cd celeritas
   $ cmake --preset=larsoft .
   $ cmake --preset=larsoft --install .

On some machines such as Perlmutter, which has Nvidia's HPC SDK installed, you
may need additional setup inside a container to configure Celeritas with CUDA:

.. code:: sh

   function export-native-cuda() {
       HPCSDK_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5"
       export CUDA_HOME="$HPCSDK_DIR/cuda/12.9"
       export PATH="$CUDA_HOME/bin":$PATH
       export CUDACXX="$CUDA_HOME/bin/nvcc"
       export CUDAARCHS=80 # For Nvidia A100

       export CPATH="$HPCSDK_DIR/math_libs/12.9/include:$CPATH"
       export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$CUDA_HOME/extras/Debugger/lib64:$HPCSDK_DIR/math_libs/12.9/lib64:$LD_LIBRARY_PATH"
   }
