.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_system:

System
======

These are low-level system options set up once per program execution, such as
enabling GPU. They are not loaded by the :cpp:struct:`Problem` definition but
are used when :ref:`api_problem_setup`.

.. celerstruct:: inp::System
.. celerstruct:: inp::Device
