.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: cmake

.. _infrastructure:

**************
Infrastructure
**************

Celeritas is built using modern CMake_. It has multiple dependencies to operate
as a full-featured code, but each dependency can be individually disabled as
needed.

.. _CMake: https://cmake.org


Installation
============

This project requires external dependencies to build with full functionality.
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


Downstream usage as a library
=============================

The Celeritas library is most easily used when your downstream app is built with
CMake. It should require a single line to initialize::

   find_package(Celeritas REQUIRED CONFIG)

and if VecGeom or CUDA are disabled a single line to link::

   target_link_libraries(mycode PUBLIC Celeritas::Core)

Because of complexities involving CUDA Relocatable Device Code, linking when
using both CUDA and VecGeom requires an additional include and the replacement
of ``target_link_libraries`` with a customized version::

  include(CeleritasLibrary)
  celeritas_target_link_libraries(mycode PUBLIC Celeritas::Core)


Developing
==========

Ideally you will build Celeritas with all dependencies to gain the full
functionality of the code, but there are circumstances in which you may not
have all the dependencies or features available. By default, the CMake code in
Celeritas queries available packages and sets several `CELERITAS_USE_{package}`
options based on what it finds, so you have a good chance of successfully
configuring Celeritas on the first go. Two optional components,
`CELERITAS_BUILD_<DEMOS|TESTS>`, will error in the configure if their required
components are missing, but they will update the CMake cache variable so that
the next configure will succeed (but with that component disabled).


CMake presets
-------------

For a slightly more advanced but potentially simpler setup, you can use the
CMake presets provided by Celeritas via the `CMakePresets.json` file for CMake
3.21 and higher:

.. code-block:: console

   $ cmake --preset=default

The three main options are "minimal", "default", and "full", which all set
different expectations for available dependencies.

If you want to add your own set of custom options and flags, create a
``CMakeUserPresets.json`` file or, if you are a developer, create a preset at
``scripts/cmake-presets/${HOSTNAME%%.*}.json`` and call ``scripts/build.sh
{preset}`` to create the symlink, configure the preset, build, and test. See
[the scripts readme](scripts/README.md) for more details.

If your CMake version is too old, you may get an unhelpful message:

.. code-block:: console

   CMake Error: Could not read presets from celeritas: Unrecognized "version"
   field

which is just a poor way of saying the version in the ``CMakePresets.json`` file
is newer than that version knows how to handle.

See the :ref:`development` section for additional development guidelines.
