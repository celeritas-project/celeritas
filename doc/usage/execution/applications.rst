.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: none

Applications
============

.. _celer-sim:

Standalone simulation app (celer-sim)
-------------------------------------

The ``celer-sim`` application is the primary means of running EM test problems
for independent validation and performance analysis. See
:ref:`example_celer_sim` for an example.

Usage:

.. literalinclude:: _usage/celer-sim.txt

- :file:`{input}.json` is the path to the input file, or ``-`` to read the
  JSON from ``stdin``.
- The ``--config`` option prints the contents of the ``["system"]["build"]``
  diagnostic output. It includes configuration options and the version number.
- The ``--device`` option prints diagnostic output for the default GPU, similar
  to the output from the ``deviceQuery`` CUDA example.
- The ``--dump-default`` option prints the default options for the execution.
  Not all variables will be shown, because some are conditional on others.

Input
^^^^^

.. todo::
   The input parameters will be documented for version 1 in the :ref:`input`
   section and :ref:`inp_standalone_input`. Until then, refer to the
   source code at :file:`app/celer-sim/RunnerInput.hh` .

In addition to these input parameters, :ref:`environment` can be specified to
change the program behavior.

Output
^^^^^^

The primary output from ``celer-sim`` is a JSON object that includes several
levels of diagnostic and result data (see :ref:`api_io`). The JSON
output should be the only data sent to ``stdout``, so it should be suitable for
piping directly into other executables such as Python or ``jq``.

Additional user-oriented output is sent to ``stderr`` via the Logger facility
(see :ref:`logging`).

.. _celer-g4:

Integrated Geant4 application (celer-g4)
----------------------------------------

The ``celer-g4`` app is a Geant4 application that offloads EM tracks to
Celeritas. It takes as input a GDML file with the detector description and
sensitive detectors marked via an ``auxiliary`` annotation. The input particles
must be specified with a HepMC3-compatible file or with a JSON-specified
"particle gun." See :ref:`example_celer_g4` for an example.

Usage:

.. literalinclude:: _usage/celer-g4.txt

Input
^^^^^

Physics is set up using the top-level ``physics_option`` key in the JSON input,
corresponding to :ref:`api_geant4_physics_options`. The magnetic field is
specified with a combination of the ``field_type``, ``field``, and
``field_file`` keys, and detailed field driver configuration options are set
with ``field_options`` corresponding to the ``FieldOptions`` class in :ref:`api_field_data`.

.. deprecated:: v0.5

   The macro file usage is in the process of being replaced by JSON
   input for improved automation. The input parameters will be documented for
   version 1 in the :ref:`input` section and
   :ref:`inp_standalone_input`.  Until then, refer to the source code
   at :file:`app/celer-g4/RunInput.hh` .

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
^^^^^^

The ROOT "MC truth" output file, if enabled with the command above, contains
hits from all the sensitive detectors.
