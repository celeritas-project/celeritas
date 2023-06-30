.. Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: none

.. _usage:

***************
Using Celeritas
***************

Celeritas includes a core set of libraries for internal and external use, as
well as several helper applications and front ends.

Software library
================

The most stable part of Celeritas is, at the present time, the high-level
program interface to the :ref:`accel` code library. However, many other
components of the API are stable and documented in the :code:`api` section.

CMake integration
-----------------

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

The :ref:`example_minimal` example demonstrates how to use Celeritas as a
library with a short standalone CMake project.

.. _celer-sim:

Standalone simulation app (celer-sim)
=====================================

The ``celer-sim`` application is the primary means of running test problems for
independent validation and performance analysis.

Usage::

   usage: celer-sim {input}.json
          celer-sim [--help|-h]
          celer-sim --version
          celer-sim --config
          celer-sim --dump-default


- :file:`{input}.json` is the path to the input file, or ``-`` to read the
  JSON from ``stdin``.
- The ``--config`` option prints the contents of the ``["system"]["build"]``
  diagnostic output. It includes configuration options and the version number.
- The ``--dump-default`` option prints the default options for the execution.
  Not all variables will be shown, because some are conditional on others.

Input
-----

.. todo::
   The input parameters will be documented for version 1.0.0. Until then, refer
   to the source code at :file:`app/celer-sim/RunnerInput.hh` .

In addition to these input parameters, :ref:`environment` can be specified to
change the program behavior.

Output
------

The primary output from ``celer-sim`` is a JSON object that includes several
levels of diagnostic and result data (see :ref:`api_io`). The JSON
output should be the only data sent to ``stdout``, so it should be suitable for
piping directly into other executables such as Python or ``jq``.

Additional user-oriented output is sent to ``stderr`` via the Logger facility
(see :ref:`logging`).

.. _celer-g4:

Integrated Geant4 application (celer-g4)
========================================

The ``celer-g4`` app is a Geant4 application that offloads EM tracks to
Celeritas. It takes as input a GDML file with the detector description and
sensitive detectors marked via an ``auxiliary`` annotation. The input particles
must be specified with a HepMC3-compatible file.

Usage::

  celer-g4 {commands}.mac

Input
-----

The input is a Geant4 macro file for executing the program. Celeritas defines
several macros in the ``/celer`` and (if CUDA is available) ``/celer/cuda/``
directories: see :ref:`api_accel_high_level` for a listing.

The ``celer-g4`` app defines several additional configuration commands under
``/celerg4``:

.. table:: Geant4 UI commands defined by ``celer-g4``.

 ============== ===============================================
 Command        Description
 ============== ===============================================
 geometryFile   Filename of the GDML detector geometry
 eventFile      Filename of the event input read by HepMC3
 rootBufferSize Buffer size of output root file [bytes]
 writeSDHits    Write a ROOT output file with hits from the SDs
 magFieldZ      Set Z-axis magnetic field strength (T)
 ============== ===============================================

In addition to these input parameters, :ref:`environment` can be specified to
change the program behavior.

Output
------

The ROOT "MC truth" output file, if enabled with the command above, contains
hits from all the sensitive detectors.

Additional utilities
====================

The Celeritas installation includes additional utilities for inspecting input
and output.

.. _celer-export-geant:

celer-export-geant
------------------

This utility exports the physics and geometry data needed to run Celeritas
without directly calling Geant4 for an independent run. Since it isolates
Celeritas from any existing Geant4 installation it can also be a means of
debugging whether a behavior change is due to a code change in Celeritas or
(for example) a change in cross sections from Geant4.

----

Usage::

   celer-export-geant {input}.gdml [{options}.json, -, ''] {output}.root
   celer-export-geant --dump-default

input
  Detector definition file

options
  An optional argument for specifying a JSON file with Geant4 setup options
  corresponding to the :ref:`api_geant4_physics_options` struct.

output
  A ROOT output file with the exported :ref:`api_importdata`.


The ``--dump-default`` usage renders the default options.


celer-dump-data
---------------

This utility prints an RST-formatted high-level dump of physics data exported
via :ref:`celer-export-geant`.

----

Usage::

   celer-dump-data {output}.root

output
  A ROOT file containing exported :ref:`api_importdata`.

.. _environment:

Environment variables
=====================

Some pieces of core Celeritas code interrogate the environment for variables to
change system- or output-level behavior. These variables are checked once per
execution, and checking them inserts the key and user-defined value (or empty)
into a diagnostic database saved to Celeritas' JSON output, so the user can
tell what variables are in use or may be useful.

.. table:: Environment variables used by Celeritas.

 ======================= ========= ==========================================
 Variable                Component Brief description
 ======================= ========= ==========================================
 CELER_BLOCK_SIZE        corecel   Change the default block size for kernels
 CELER_COLOR             corecel   Enable/disable ANSI color logging
 CELER_DEBUG_DEVICE      corecel   Increase device error checking and output
 CELER_DISABLE_DEVICE    corecel   Disable CUDA/HIP support
 CELER_DISABLE_PARALLEL  corecel   Disable MPI support
 CELER_ENABLE_PROFILING  corecel   Set up NVTX profiling ranges
 CELER_LOG               corecel   Set the "global" logger verbosity
 CELER_LOG_LOCAL         corecel   Set the "local" logger verbosity
 CELER_PROFILE_DEVICE    corecel   Record extra kernel launch information
 CUDA_STACK_SIZE         celeritas Change ``cudaLimitStackSize`` for VecGeom
 CUDA_HEAP_SIZE          celeritas Change ``cudaLimitMallocHeapSize`` (VG)
 G4VG_COMPARE_VOLUMES    celeritas Check G4VG volume capacity when converting
 CELER_DISABLE           accel     Disable Celeritas offloading entirely
 CELER_STRIP_SOURCEDIR   accel     Strip directories from exception output
 ======================= ========= ==========================================

Environment variables from external libraries can also be referenced by
Celeritas or its apps:

.. table:: Environment variables used by relevant external libraries.

 ======================== ========= ==========================================
 Variable                 Library   Brief description
 ======================== ========= ==========================================
 CUDA_VISIBLE_DEVICES     CUDA      Set the active CUDA device
 G4LEDATA                 Geant4    Path to low-energy EM data
 G4FORCE_RUN_MANAGER_TYPE Geant4    Use MT or Serial thread layout
 G4FORCENUMBEROFTHREADS   Geant4    Set CPU worker thread count
 OMP_NUM_THREADS          OpenMP    Number of threads per process
 ======================== ========= ==========================================

.. _logging:

Logging
=======

The Celeritas library writes informational messages to ``stderr``. The given
levels can be used with the ``CELER_LOG`` and ``CELER_LOG_LOCAL`` environment
variables to suppress or increase the output. The default is to print
diagnostic messages and higher.

.. table:: Logging levels in increasing severity.

 ========== ==============================================================
 Level      Description
 ========== ==============================================================
 debug      Low-level debugging messages
 diagnostic Diagnostics about current program execution
 status     Program execution status (what stage is beginning)
 info       Important informational messages
 warning    Warnings about unusual events
 error      Something went wrong, but execution can continue
 critical   Something went terribly wrong, program termination imminent
 ========== ==============================================================

