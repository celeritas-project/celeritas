.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: console

.. _profiling:

Performance profiling
=====================

Since the primary motivator of Celeritas is performance on GPU hardware,
profiling is a necessity. Celeritas uses NVTX (CUDA),  ROCTX (HIP) or Perfetto (CPU)
to annotate the different sections of the code, allowing for fine-grained
profiling and improved visualization.

Tracing events in Celeritas
---------------------------

Celeritas includes a number of NVTX, HIP, and Perfetto events that can be used to
trace different aspects of the code execution. These events are enabled
when the environment variable ``CELER_ENABLE_PROFILING`` (see :ref:`environment`) is set.
All profiling backends (CUDA, HIP, and Perfetto)
support both Timeline and Counter events detailed below, except that HIP does not support Counters.

Profiling backends allow grouping various events into "namespaces" (NVTX/HIP domains, Perfetto categories) so that users can selectively enable events they are interested in. Celeritas groups all events in the "celeritas" namespace.

Slices
^^^^^^
Detailed timing of each step iteration is recorded with "slices" events in Celeritas. The step slice contains nested Slices
for each action composing the step, some actions such as along-step actions contain more nested slices for fine-grained profiling.

In addition to the slices in the simulation loop, slices events are also recorded when setting up the problem (e.g. detector construction)

Counters
^^^^^^^^
Celeritas provides a few counter events. Currently it writes:

- active, alive, and dead track counts at each step iteration, and
- the number of hits in a step.

Profiling a Celeritas example app
---------------------------------

A detailed timeline of the Celeritas construction, steps, and kernel launches
can be gathered, the example below illustrates how to do it using `NVIDIA Nsight systems`_.

.. _NVIDIA Nsight systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html

Here is an example invoking the ``celer-sim`` app through the NVIDIA utility to
generate a timeline:

.. sourcecode::
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > nsys profile \
   >   --trace=cuda,nvtx,osrt \
   >   --nvtx-capture="celeritas" \
   >   --osrt-backtrace-stack-size=16384 --backtrace=fp \
   >   -o report.nsys-rep -f true \
   >   celer-sim inp.json

Line 2 specifies the APIs to be captured: in this case, CUDA calls, NVTX
ranges, and OS runtime libraries.
To use the NVTX ranges, which is strongly recommended for kernel annotations,
you **must** enable the ``CELER_ENABLE_PROFILING`` variable
in addition to using the NVTX "trace" option (line 3).
The capture domain in line 4 restricts profiling to the Celeritas application.
Additional frame-pointer-based backtracing is specified in line 5;
line 6 writes (and overwrites) to a particular output file;
the final line invokes the application.

On AMD hardware using the ROCProfiler_, here's an example that writes out timeline information:

.. sourcecode::
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

On CPU, timelines are generated using Perfetto, which is only supported when CUDA
and HIP are disabled. Perfetto supports application-level and system-level profiling.
To use the application-level profiling, see :ref:`inp_diagnostics`.

.. sourcecode::
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > celer-sim inp.json

The system-level profiling, capturing both system and application events,
requires starting external services. Details on how to setup the system services can be found in
the `Perfetto documentation`_. Root access on the system is required.

Integration with user applications
----------------------------------

When using a CUDA or HIP backend, **no additional code is needed in the user
application**.
The commands shown in the previous sections can be used to profile your application.
If your application already uses NVTX, or ROCTX, you can exclude Celeritas events by excluding the ``celeritas`` domain.

When using Perfetto for CPU profiling, you need to create a
:cpp:class:celeritas::TracingSession:
instance. The profiling session needs to be explicitly started, and will end when the object goes out of scope,
but it can be moved to extend its lifetime.

.. sourcecode:: cpp
   :linenos:

   #include "corecel/sys/TracingSession.hh"

   int main()
   {
      // Pass with empty/no filename for system profiling
      celeritas::TracingSession session("out.perfetto");
   }

As mentioned above, Perfetto can either profile application events only, or application and system events.
The system-level profiling requires starting external services. Details on how to setup the system services can be found in the `Perfetto documentation`_. Root access on the system is required.

When the tracing session is started with a filename, the application-level profiling is used and written to the specified file.
Omitting the filename will use the system-level profiling, in which case you must have the external Perfetto tracing processes started. The container in ``scripts/docker/interactive`` provides an example Perfetto configuration for tracing both system-level and Celeritas events.

As with NVTX and ROCTX, if your application already uses Perfetto, you can exclude Celeritas events by excluding the ``celeritas`` category.

.. _Perfetto documentation: https://perfetto.dev/docs/quickstart/linux-tracing

Kernel profiling
----------------

Detailed kernel diagnostics including occupancy and memory bandwidth can be
gathered with the `NVIDIA Nsight Compute`_ profiler.

.. _NVIDIA Nsight Compute: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

The ``Source`` tab is especially useful, as it maps diagnostics such as number
of instructions executed and number of stalls to individual lines in the source
code. To obtain this mapping, Celeritas must be compiled with ``-lineinfo`` in
``CMAKE_CUDA_FLAGS``. The following ``ncu`` command can be run to obtain an
``.ncu-rep`` for use with NVIDIA Nsight Compute.

.. sourcecode::
   :linenos:

    $ ncu \
    > --set full \
    > --kernel-name launch_action_impl \
    > --launch-skip 254 --launch-count 1 \
    > -f -o output \
    > celer-sim inp.json

The metric set must be ``full`` for detailed source-level profiling
information.  Note that full metrics require 30+ passes and thus are very
time-consuming; use ``basic`` to rapidly obtain coarse-grained profiling
information.  This command will force-overwrite an output file named
``output.ncu-rep``.  In this example, only a single kernel is profiled: the
255th launch of the ``launch_action_impl``. Selecting how many kernels must be
skipped before reaching the kernel of interest can be done in NVIDIA Nsight
Systems prior to running NVIDIA Nsight Compute. This can be done in two ways. The
first method is:

1. Click ``CUDA HW`` then ``[All Streams]`` then ``Kernels``
2. Right-click ``launch_action_impl`` and click ``Show in Events View``
3. Select any kernel in the event view
4. Obtain the launch number of the kernel in the ``#`` column
5. Use ``#`` minus 1 as the ``launch-skip``

This method can be used to profile the most expensive kernel by sorting the
event view by duration and finding the ``#`` of the top kernel.  The second
method is:

1. Right-click the kernel of interest in the timeline
2. Click ``Analyze the selected kernel with NVIDIA Nsight Compute``
3. Click ``Display the command line to use NVIDIA Nsight Compute CLI``
4. Copy the ``launch-skip`` from the supplied ``ncu`` command.

A final option is to profile a kernel using NVTX annotations:

.. sourcecode::
   :linenos:

   $ CELER_ENABLE_PROFILING=1 \
   > ncu \
   > --set full \
   > --nvtx --nvtx-include "celeritas@{RANGE}" \
   > --launch-skip 100 --launch-count 1 \
   > -f -o output \
   > celer-sim inp.json

Here, :samp:`celeritas@{RANGE}` is the name of the NVTX domain and
slash-separated range, (e.g. ``run/step/*/propagate``).
Note that the domain and range are flipped compared to ``nsys`` since the
kernel profiling allows detailed top-down stack specification.

Using Instruments on macOS
--------------------------

Celeritas developers on macOS can use Apple's Instruments_ code to analyze CPU
performance.
Running against a locally-built app for the first time will usually fail with
an error "Failed to attach to target process."
This is because the app built with standard CMake tools does not have a code
signature with the necessary entitlements.
When building Celeritas apps on an Apple machine, the CMake option
``CELERITAS_CODESIGN_APPS`` becomes available.
Enabling this option will automatically call ``codesign`` after building each
app executable, which will allow its selection in Instruments.


.. _Instruments: https://developer.apple.com/tutorials/instruments
