.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _environment:

Environment variables
=====================

.. note:: Many of these environment variables will eventually be replaced by
   :ref:`inp_control` and :ref:`inp_system` options in Celeritas v1.0 and beyond.

Some pieces of core Celeritas code interrogate the environment for variables to
change system- or output-level behavior. These variables are checked once per
execution, and checking them inserts the key and user-defined value (or empty)
into a diagnostic database saved to Celeritas' JSON output, so the user can
tell what variables are in use or may be useful. A variable marked as a *flag*
has a default value but can be overridden with any boolean value.

.. table:: Environment variables used by Celeritas.

 ========================= ========= ==========================================
 Variable                  Component Brief description
 ========================= ========= ==========================================
 CELER_COLOR               corecel   Flag: log with ANSI colors
 CELER_DEBUG_DEVICE        corecel   Flag: check device ID consistency
 CELER_DISABLE_DEVICE      corecel   Flag: disable CUDA/HIP support
 CELER_DISABLE_PARALLEL    corecel   Flag: disable MPI support
 CELER_DISABLE_REDIRECT    corecel   Flag: disable stream->logger redirection
 CELER_DISABLE_ROOT        corecel   Flag: disable ROOT I/O calls
 CELER_DISABLE_SIGNALS     corecel   Flag: disable signal handling
 CELER_DEVICE_ASYNC        corecel   Flag: allocate memory asynchronously
 CELER_ENABLE_PROFILING    corecel   Flag: use NVTX/ROCTX profiling [#pr]_
 CELER_LOG                 corecel   Set "global" logger verbosity [#lg]_
 CELER_LOG_LOCAL           corecel   Set "local" logger verbosity [#lg]_
 CELER_MEMPOOL... [#mp]_   corecel   Set ``cudaMemPoolAttrReleaseThreshold``
 CELER_PERFETT... [#bs]_   corecel   Set in-process tracing buffer size
 CELER_PROFILE_DEVICE      corecel   Flag: record kernel launch counts
 CUDA_HEAP_SIZE            geocel    Set ``cudaLimitMallocHeapSize`` (VG)
 CUDA_STACK_SIZE           geocel    Set ``cudaLimitStackSize`` for VecGeom
 G4ORG_OPTIONS             orange    JSON filename for G4-to-ORANGE conversion
 G4VG_COMPARE_VOLUMES      geocel    Check G4VG volume capacity when converting
 HEPMC3_VERBOSE            celeritas HepMC3 debug level integer
 VECGEOM_VERBOSE           celeritas VecGeom CUDA verbosity integer
 CELER_DISABLE             accel     Flag: disable Celeritas offloading entirely
 CELER_LOG_ALL_LOCAL       accel     Flag: log more messages from all threads
 CELER_KILL_OFFLOAD        accel     Flag: kill offloaded tracks [#ko]_
 CELER_NONFATAL_FLUSH      accel     Flag: print and continue on failure [#nf]_
 CELER_STRIP_SOURCEDIR     accel     Flag: clean exception output
 ORANGE_BIH_MAX_LEAF_SIZE  orange    Set BIH ``max_leaf_size``, i.e., the
                                     maximum number of bboxes that can reside on
                                     a leaf node without triggering a
                                     partitioning attempt
 ORAMGE_BIH_DEPTH_LIMIT    orange    Set BIH ``depth_limit``, i.e., the hard
                                     limit on the depth of most the embedded
                                     node (where 1 is the root node)
 ORANGE_BIH_PART_CANDS     orange    Set BIH ``num_part_cands``, i.e., the
                                     number of partition candidates to check per
                                     axis when partitioning a node during BIH
                                     construction
 ORANGE_BIH_STRUCTURE      orange    Include "structure" info in BIH JSON output
 ========================= ========= ==========================================

.. [#pr] See :ref:`profiling`
.. [#lg] See :ref:`logging`
.. [#mp] CELER_MEMPOOL_RELEASE_THRESHOLD changes the amount of memory used for
   the device asynchronous memory pool.
.. [#bs] CELER_PERFETTO_BUFFER_SIZE_MB
.. [#ko] This flag, used by ``accel``, omits Celeritas setup and instead kills
   tracks that would have otherwise been offloaded. This capability can be used
   to estimate a maximum speedup by using Celeritas.
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

.. doxygenfunction:: celeritas::use_color

.. _logging:

Logging
-------

The Celeritas library has an internal logger that, by default, writes
informational messages to ``stderr`` through the ``std::clog`` (buffered
standard error) handle.
The "world" logger generally prints messages during setup or at circumstances
shared by all threads/tasks/processes, and the "self" logger is for messages
related to a particular thread. By default the self logger is mutexed for
thread safety, and log messages are assembled internally before printing to
reduce I/O contention.

The levels below can be used with the ``CELER_LOG`` and ``CELER_LOG_LOCAL``
environment variables to suppress or increase the output.
The default is to print diagnostic messages and higher.

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
