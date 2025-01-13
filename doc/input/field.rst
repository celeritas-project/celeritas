.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_field:

*****
Field
*****

The field currently allows a few hardcoded options. It will be extended to
additional field types and may allow completely custom field implementations.

.. doxygenstruct:: celeritas::inp::NoField
.. doxygenstruct:: celeritas::inp::UniformField
.. doxygentypedef:: celeritas::inp::RZMapField

The field type is selected with a variant:

.. doxygentypedef:: celeritas::inp::Field

