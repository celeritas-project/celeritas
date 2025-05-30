.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_field:

Field
=====

The field currently allows a few hard-coded options. It will be extended to
additional field types and may allow completely custom field implementations.

.. doxygenstruct:: celeritas::inp::NoField
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::UniformField
   :members:
   :no-link:

.. doxygentypedef:: celeritas::inp::RZMapField

The field type is selected with a variant:

.. doxygentypedef:: celeritas::inp::Field
