.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_field:

Field
=====

The field type is selected with a variant:

.. doxygentypedef:: celeritas::inp::Field

The field currently allows a few hard-coded options. It will be extended to
additional field types and may allow completely custom field implementations.

.. celerstruct:: inp::NoField
.. celerstruct:: inp::UniformField
.. celerstruct:: inp::CylMapField
.. celerstruct:: inp::CartMapField
.. celerstruct:: RZMapFieldInput

The field driver options are not yet a stable part of the API:

.. celerstruct:: FieldDriverOptions
