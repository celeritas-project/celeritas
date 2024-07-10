.. Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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

and if VecGeom or CUDA are disabled, a single line to link::

   target_link_libraries(mycode PUBLIC Celeritas::celeritas)

Because of complexities involving CUDA Relocatable Device Code (RDC), consuming
Celeritas with CUDA and VecGeom support requires an additional include and the
use of wrappers around CMake's target commands::

  include(CudaRdcUtils)
  cuda_rdc_add_executable(mycode ...)
  cuda_rdc_target_link_libraries(mycode PUBLIC Celeritas::celeritas)

As the ``cuda_rdc_...`` functions decay to the wrapped CMake commands if CUDA
and VecGeom are disabled, you can use them to safely build and link nearly all targets
consuming Celeritas in your project. This provides tracking of the appropriate
sequence of linking for the final application whether it uses CUDA code or not,
and whether Celeritas is CPU-only or CUDA enabled::

  cuda_rdc_add_library(myconsumer SHARED ...)
  cuda_rdc_target_link_libraries(myconsumer PUBLIC Celeritas::celeritas)

  cuda_rdc_add_executable(myapplication ...)
  cuda_rdc_target_link_libraries(myapplication PRIVATE myconsumer)

If your project builds shared libraries that are intended to be loaded at
application runtime (e.g. via ``dlopen``), you should prefer use the CMake
``MODULE`` target type::

  cuda_rdc_add_library(myplugin MODULE ...)
  cuda_rdc_target_link_libraries(myplugin PRIVATE Celeritas::celeritas)

This is recommended as ``cuda_rdc_target_link_libraries`` understands these as
a final target for which all device symbols require resolving. If you are
forced to use the ``SHARED`` target type for plugin libraries (e.g. via your
project's own wrapper functions), then these should be declared with the CMake
or project-specific commands with linking to both the primary Celeritas target
and its device code counterpart::

  add_library(mybadplugin SHARED ...)
  # ... or myproject_add_library(mybadplugin ...)
  target_link_libraries(mybadplugin PRIVATE Celeritas::celeritas $<TARGET_NAME_IF_EXISTS:Celeritas::celeritas_final>)
  # ... or otherwise declare the plugin as requiring linking to the two targets

Celeritas device code counterpart target names are always the name of the
primary target appended with ``_final``. They are only present if Celeritas was
built with CUDA support so it is recommended to use the CMake generator
expression above to support CUDA or CPU-only builds transparently.

The :ref:`example_minimal` example demonstrates how to use Celeritas as a
library with a short standalone CMake project.

.. _celer-sim:

Standalone simulation app (celer-sim)
=====================================

The ``celer-sim`` application is the primary means of running EM test problems
for independent validation and performance analysis.

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
must be specified with a HepMC3-compatible file or with a JSON-specified
"particle gun."

Usage::

  celer-g4 {input}.json
           {commands}.mac
           --interactive
           --dump-default

Input
-----

Physics is set up using the top-level ``physics_option`` key in the JSON input,
corresponding to :ref:`api_geant4_physics_options`. The magnetic field is
specified with a combination of the ``field_type``, ``field``, and
``field_file`` keys, and detailed field driver configuration options are set
with ``field_options`` corresponding to the ``FieldOptions`` class in :ref:`api_field_data`.

.. note:: The macro file usage is in the process of being replaced by JSON
   input for improved automation.

The input is a Geant4 macro file for executing the program. Celeritas defines
several macros in the ``/celer`` and (if CUDA is available) ``/celer/cuda/``
directories: see :ref:`api_accel_high_level` for a listing.

The ``celer-g4`` app defines several additional configuration commands under
``/celerg4``:

.. table:: Geant4 UI commands defined by ``celer-g4``.

 ================== ==================================================
 Command            Description
 ================== ==================================================
 geometryFile       Filename of the GDML detector geometry
 eventFile          Filename of the event input read by HepMC3
 rootBufferSize     Buffer size of output root file [bytes]
 writeSDHits        Write a ROOT output file with hits from the SDs
 stepDiagnostic     Collect the distribution of steps per Geant4 track
 stepDiagnosticBins Number of bins for the Geant4 step diagnostic
 fieldType          Select the field type [rzmap|uniform]
 fieldFile          Filename of the rz-map loaded by RZMapFieldInput
 magFieldZ          Set Z-axis magnetic field strength (T)
 ================== ==================================================

In addition to these input parameters, :ref:`environment` can be specified to
change the program behavior.

Output
------

The ROOT "MC truth" output file, if enabled with the command above, contains
hits from all the sensitive detectors.


.. _celer-geo:

Visualization application (celer-geo)
=====================================

The ``celer-geo`` app is a server-like front end to the Celeritas geometry
interfaces that can generate exact images of a user geometry model.

Usage::

  celer-geo {input}.jsonl
            -

Input
-----

The input and output are both formatted as `JSON lines`_, a format where each
line (i.e., text ending with ``\\n``) is a valid JSON object. Each line of
input executes a command in ``celer-geo`` which will print to ``stdout`` a
single JSON line. Log messages are sent to ``stderr`` and can be
controlled by the :ref:`environment` variables.

The first input command must define the input model (and may define additional
device settings)::

   {"geometry_file": "simple-cms.gdml"}

Subsequent lines will each specify the imaging window, the geometry, the
binary image output filename, and the execution space (device or host for GPU
or CPU, respectively).::

   {"image": {"_units": "cgs", "lower_left": [-800, 0, -1500], "upper_right": [800, 0, 1600], "rightward": [1, 0, 0], "vertical_pixels": 128}, "volumes": true, "bin_file": "simple-cms-cpu.orange.bin"}

After the first image window is specified, it will be reused if the "image" key
is omitted. A new geometry and/or execution space may be specified, useful for
verifying different navigators behave identically::

   {"bin_file": "simple-cms-cpu.geant4.bin", "geometry": "geant4"}

An interrupt signal (``^C``), end-of-file (``^D``), or empty command will all
terminate the server.

.. _JSON lines: https://jsonlines.org

Output
------

If an input command is invalid or empty, an "example" (i.e., default but
incomplete input) will be output and the program may continue or be terminated.

A successful raytrace will print the actually-used image parameters, geometry,
and execution space. If the "volumes" key was set to true, it will also
determine and print all the volume names for the geometry.

When the server is directed to terminate, it will print diagnostic information
about the code, including timers about the geometry loading and tracing.

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


orange-update
-------------

Read an ORANGE JSON input file and write it out again. This is used for
updating from an older version of the input (i.e. with different parameter
names or fewer options) to a newer version.

----

Usage::

   orange-update {input}.org.json {output}.org.json

Either of the filenames can be replaced by ``-`` to read from stdin or write to
stdout.


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
 CELER_COLOR             corecel   Enable/disable ANSI color logging
 CELER_DEBUG_DEVICE      corecel   Increase device error checking and output
 CELER_DISABLE_DEVICE    corecel   Disable CUDA/HIP support
 CELER_DISABLE_PARALLEL  corecel   Disable MPI support
 CELER_DISABLE_ROOT      corecel   Disable ROOT I/O calls
 CELER_ENABLE_PROFILING  corecel   Set up NVTX/ROCTX profiling ranges [#pr]
 CELER_LOG               corecel   Set the "global" logger verbosity
 CELER_LOG_LOCAL         corecel   Set the "local" logger verbosity
 CELER_MEMPOOL... [#mp]_ corecel   Change ``cudaMemPoolAttrReleaseThreshold``
 CELER_PERFETT... [#bs]_ corecel   Set the in-process tracing buffer size
 CELER_PROFILE_DEVICE    corecel   Record extra kernel launch information
 DEVICE_DISABLE_ASYNC    corecel   Disable asyncronous memory allocation
 CUDA_HEAP_SIZE          geocel    Change ``cudaLimitMallocHeapSize`` (VG)
 CUDA_STACK_SIZE         geocel    Change ``cudaLimitStackSize`` for VecGeom
 G4VG_COMPARE_VOLUMES    geocel    Check G4VG volume capacity when converting
 HEPMC3_VERBOSE          celeritas HepMC3 debug verbosity
 VECGEOM_VERBOSE         celeritas VecGeom CUDA verbosity
 CELER_DISABLE           accel     Disable Celeritas offloading entirely
 CELER_KILL_OFFLOAD      accel     Kill Celeritas-supported tracks in Geant4
 CELER_STRIP_SOURCEDIR   accel     Strip directories from exception output
 ======================= ========= ==========================================

.. [#bs] CELER_PERFETTO_BUFFER_SIZE_MB
.. [#mp] CELER_MEMPOOL_RELEASE_THRESHOLD
.. [#pr] See :ref:`profiling`

Some of the Celeritas-defined environment variables have prefixes from other
libraries because they directly control the behavior of that library and
nothing else. The ``DEVICE_DISABLE_ASYNC`` may be needed when running HIP 5.7
or later due to the "beta" nature of hipMallocAsync_.

.. _hipMallocAsync: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_o.html

Environment variables from external libraries can also be referenced by
Celeritas or its apps:

.. table:: Environment variables used by relevant external libraries.

 ======================== ========= ==========================================
 Variable                 Library   Brief description
 ======================== ========= ==========================================
 CUDA_VISIBLE_DEVICES     CUDA      Set the active CUDA device
 HIP_VISIBLE_DEVICES      HIP       Set the active HIP device
 G4LEDATA                 Geant4    Path to low-energy EM data
 G4FORCE_RUN_MANAGER_TYPE Geant4    Use MT or Serial thread layout
 G4FORCENUMBEROFTHREADS   Geant4    Set CPU worker thread count
 OMP_NUM_THREADS          OpenMP    Number of threads per process
 ======================== ========= ==========================================

.. note::

   For frameworks integrating Celeritas, these options are configurable via the
   Celeritas API. Before Celeritas is set up for the first time, on a single
   thread access the ``celeritas::environment()`` struct (see
   :ref:`api_system`), and call ``insert`` for the desired key/value pairs.

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


.. _profiling:

Profiling
=========

Since the primary motivator of Celeritas is performance on GPU hardware,
profiling is a necessity. Celeritas uses NVTX (CUDA),  ROCTX (HIP) or Perfetto (CPU)
to annotate the different sections of the code, allowing for fine-grained
profiling and improved visualization.

Timelines
---------

A detailed timeline of the Celeritas construction, steps, and kernel launches
can be gathered using `NVIDIA Nsight systems`_.

.. _NVIDIA Nsight systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html

Here is an example using the ``celer-sim`` app to generate a timeline:

.. sourcecode:: console
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > nsys profile \
   > -c nvtx  --trace=cuda,nvtx,osrt
   > -p celer-sim@celeritas
   > --osrt-backtrace-stack-size=16384 --backtrace=fp
   > -f true -o report.qdrep \
   > celer-sim inp.json

To use the NVTX ranges, you must enable the ``CELER_ENABLE_PROFILING`` variable
and use the NVTX "capture" option (lines 1 and 3). The ``celer-sim`` range in
the ``celeritas`` domain (line 4) enables profiling over the whole application.
Additional system backtracing is specified in line 5; line 6 writes (and
overwrites) to a particular output file; the final line invokes the
application.

Timelines can also be generated on AMD hardware using the ROCProfiler_
applications. Here's an example that writes out timeline information:

.. sourcecode:: console
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > rocprof \
   > --roctx-trace \
   > --hip-trace \
   > celer-sim inp.json

.. _ROCProfiler: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprofv1.html#roctx-trace

It will output a :file:`results.json` file that contains profiling data for
both the Celeritas annotations (line 3) and HIP function calls (line 4) in
a "trace event format" which can be viewed in the Perfetto_ data visualization
tool.

.. _Perfetto: https://ui.perfetto.dev/

On CPU, timelines are generated using Perfetto. It is only supported when CUDA
and HIP are disabled. Perfetto supports application-level and system-level profiling.
To use the application-level profiling, set the ``tracing_file`` input key.

.. sourcecode:: console
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > celer-sim inp.json

The system-level profiling, capturing both system and application events,
requires starting external services. To use this mode, the ``tracing_file`` key must
be absent or empty. Details on how to setup the system services can be found in
the `Perfetto documentation`_. Root access on the system is required.

If you integrate celeritas in your application, you need to create a ``TracingSession``
instance. The profiling session will end when the object goes out of scope but it can be
moved to extend its lifetime.

.. sourcecode:: cpp
   :linenos:

   #include "TracingSession.hh"

   int main()
   {
      // system-level profiling, pass a filename to use application-level profiling
      TracingSession session;
      session.start()
   }

.. _Perfetto documentation: https://perfetto.dev/docs/quickstart/linux-tracing

Kernel profiling
----------------

Detailed kernel diagnostics including occupancy and memory bandwidth can be
gathered with the `NVIDIA Compute systems`_ profiler.

.. _NVIDIA Compute systems: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

This example gathers kernel statistics for 10 "propagate" kernels (for both
charged and uncharged particles) starting with the 300th launch.

.. sourcecode:: console
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > ncu \
   > --nvtx --nvtx-include "celeritas@celer-sim/step/*/propagate" \
   > --launch-skip 300 --launch-count 10 \
   > -f -o propagate
   > celer-sim inp.json

It will write to :file:`propagate.ncu-rep` output file. Note that the domain
and range are flipped compared to ``nsys`` since the kernel profiling allows
detailed top-down stack specification.
