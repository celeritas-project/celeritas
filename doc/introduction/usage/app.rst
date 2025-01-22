.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: none

.. _celer-sim:

Standalone simulation app (celer-sim)
-------------------------------------

The ``celer-sim`` application is the primary means of running EM test problems
for independent validation and performance analysis. See
:ref:`example_celer_sim` for an example.

Usage::

   usage: celer-sim {input}.json
          celer-sim [--help|-h]
          celer-sim --version
          celer-sim --config
          celer-sim --device
          celer-sim --dump-default


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
   The input parameters will be documented for version 1.0.0. Until then, refer
   to the source code at :file:`app/celer-sim/RunnerInput.hh` .

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

Usage::

  celer-g4 {input}.json
           {commands}.mac
           --interactive
           --dump-default

Input
^^^^^

Physics is set up using the top-level ``physics_option`` key in the JSON input,
corresponding to :ref:`api_geant4_physics_options`. The magnetic field is
specified with a combination of the ``field_type``, ``field``, and
``field_file`` keys, and detailed field driver configuration options are set
with ``field_options`` corresponding to the ``FieldOptions`` class in :ref:`api_field_data`.

.. deprecated:: The macro file usage is in the process of being replaced by JSON
   input for improved automation.  Until then, refer
   to the source code at :file:`app/celer-g4/RunInput.hh` .

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


.. _celer-geo:

Visualization application (celer-geo)
-------------------------------------

The ``celer-geo`` app is a server-like front end to the Celeritas geometry
interfaces that can generate exact images of a user geometry model.
See :ref:`example_celer_geo` for an example.

Usage::

  celer-geo {input}.jsonl
            -

Input
^^^^^

.. highlight:: json

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
^^^^^^

If an input command is invalid or empty, an "example" (i.e., default but
incomplete input) will be output and the program may continue or be terminated.

A successful raytrace will print the actually-used image parameters, geometry,
and execution space. If the "volumes" key was set to true, it will also
determine and print all the volume names for the geometry.

When the server is directed to terminate, it will print diagnostic information
about the code, including timers about the geometry loading and tracing.

Additional utilities
--------------------

The Celeritas installation includes additional utilities for inspecting input
and output.

.. _celer-export-geant:

celer-export-geant
^^^^^^^^^^^^^^^^^^

.. highlight:: none

This utility exports the physics and geometry data used to run Celeritas. It
can be used in one of two modes:

1. Export serialized data as a ROOT file to be used on a subsequent run
   of Celeritas. Since it isolates Celeritas from any existing Geant4
   installation it can also be a means of debugging whether a behavior change
   is due to a code change in Celeritas or (for example) a change in cross
   sections from Geant4.
2. Export serialized data as a JSON file for data exploration. This is a means
   to verify or plot the cross sections, volumes, etc. used by Celeritas.

----

Usage::

   celer-export-geant {input}.gdml [{options}.json, -, ''] {output}.[root, json]
   celer-export-geant --dump-default

input
  Detector definition file

options
  An optional argument for specifying a JSON file with Geant4 setup options
  corresponding to the :ref:`api_geant4_physics_options` struct.

output
  A ROOT/JSON output file with the exported :ref:`api_importdata`.


The ``--dump-default`` usage renders the default options.


celer-dump-data
^^^^^^^^^^^^^^^

This utility prints an RST-formatted high-level dump of physics data exported
via :ref:`celer-export-geant`.

----

Usage::

   celer-dump-data {output}.root

output
  A ROOT file containing exported :ref:`api_importdata`.


orange-update
^^^^^^^^^^^^^

Read an ORANGE JSON input file and write it out again. This is used for
updating from an older version of the input (i.e. with different parameter
names or fewer options) to a newer version.

----

Usage::

   orange-update {input}.org.json {output}.org.json

Either of the filenames can be replaced by ``-`` to read from stdin or write to
stdout.

