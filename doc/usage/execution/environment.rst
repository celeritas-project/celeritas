.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

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
 CELER_DEVICE_ASYNC      corecel   Flag for asynchronous memory allocation
 CELER_ENABLE_PROFILING  corecel   Set up NVTX/ROCTX profiling ranges [#pr]_
 CELER_LOG               corecel   Set the "global" logger verbosity
 CELER_LOG_LOCAL         corecel   Set the "local" logger verbosity
 CELER_MEMPOOL... [#mp]_ corecel   Change ``cudaMemPoolAttrReleaseThreshold``
 CELER_PERFETT... [#bs]_ corecel   Set the in-process tracing buffer size
 CELER_PROFILE_DEVICE    corecel   Record extra kernel launch information
 CUDA_HEAP_SIZE          geocel    Change ``cudaLimitMallocHeapSize`` (VG)
 CUDA_STACK_SIZE         geocel    Change ``cudaLimitStackSize`` for VecGeom
 G4VG_COMPARE_VOLUMES    geocel    Check G4VG volume capacity when converting
 HEPMC3_VERBOSE          celeritas HepMC3 debug verbosity
 VECGEOM_VERBOSE         celeritas VecGeom CUDA verbosity
 CELER_DISABLE           accel     Disable Celeritas offloading entirely
 CELER_KILL_OFFLOAD      accel     Kill Celeritas-supported tracks in Geant4
 CELER_NONFATAL_FLUSH    accel     Instead of crashing, kill tracks [#nf]_
 CELER_STRIP_SOURCEDIR   accel     Strip directories from exception output
 ======================= ========= ==========================================

.. [#bs] CELER_PERFETTO_BUFFER_SIZE_MB
.. [#mp] CELER_MEMPOOL_RELEASE_THRESHOLD
.. [#pr] See :ref:`profiling`
.. [#nf] Normally, exceeding the "maximum steps" or interrupting the stepping
   loop will call G4Exception, which normally kills the code. (In external
   frameworks this usually causes a stack trace and core dump.) Instead of
   doing that, kill all the active tracks and print their state. If more tracks
   are buffered, those will continue to transport.

Some of the Celeritas-defined environment variables have prefixes from other
libraries because they directly control the behavior of that library and
nothing else. The ``CELER_DEVICE_ASYNC`` may be needed when running HIP 5.7
or later due to the "beta" nature of hipMallocAsync_: it defaults to "true"
*except* for HIP less than 5.2 (where it is not implemented) or greater than 5.6.

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
-------

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
 critical   Something went terribly wrong: program termination imminent
 ========== ==============================================================

