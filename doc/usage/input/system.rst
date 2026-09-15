.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_system:

System
======

Some low-level system options, such as enabling the GPU, are set up once per
program execution. They are not loaded by the :cpp:struct:`Problem` definition
but are used by the both framework integrations and standalone applications.

.. celerstruct:: inp::System
.. celerstruct:: inp::Device
