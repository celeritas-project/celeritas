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
limited development on personal machines with fewer available components.

- [CUDA](https://developer.nvidia.com/cuda-toolkit): on-device computation
- an MPI implementation (such as [Open MPI](https://www.open-mpi.org)): shared-memory parallelism
- [ROOT](https://root.cern): I/O
- [nljson](https://github.com/nlohmann/json): simple text-based I/O for
  diagnostics and program setup
- [VecGeom](https://gitlab.cern.ch/VecGeom/VecGeom): on-device navigation of GDML-defined detector geometry
- [Geant4](https://geant4.web.cern.ch/support/download): preprocessing physics data for a problem input
- [G4EMLOW](https://geant4.web.cern.ch/support/download): EM physics model data
- [HepMC3](http://hepmc.web.cern.ch/hepmc/): Event input
- [SWIG](http://swig.org): limited set of Python wrappers for analyzing input
  data

Build/test dependencies are:

- [CMake](https://cmake.org): build system
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html): formatting enforcement
- [GoogleTest](https://github.com/google/googletest): test harness


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

See the :ref:`development` section for additional development guidelines.
