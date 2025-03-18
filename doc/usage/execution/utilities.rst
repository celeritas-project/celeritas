.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: none

Utilities
=========

.. _celer-geo:

Visualization application (celer-geo)
-------------------------------------

The ``celer-geo`` app is a server-like front end to the Celeritas geometry
interfaces that can generate exact images of a user geometry model. It should
be invoked only as part of the celerpy_ python app.  See :ref:`example_celer_geo` for an example.

Usage:

.. literalinclude:: _usage/celer-geo.txt

.. _celerpy: https://github.com/celeritas-project/celerpy

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

Usage:

.. literalinclude:: _usage/celer-export-geant.txt


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

Usage:

.. literalinclude:: _usage/celer-dump-data.txt


output
  A ROOT file containing exported :ref:`api_importdata`.


orange-update
^^^^^^^^^^^^^

Read an ORANGE JSON input file and write it out again. This is used for
updating from an older version of the input (i.e. with different parameter
names or fewer options) to a newer version.

----

Usage:

.. literalinclude:: _usage/orange-update.txt

Either of the filenames can be replaced by ``-`` to read from stdin or write to
stdout.

