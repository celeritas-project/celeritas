.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_framework:

User application and framework integration
==========================================

When integrating with user applications or frameworks, ``FrameworkInput`` is
used to define the system configuration and select the Celeritas physics to
enable (via :cpp:struct:`PhysicsFromGeant`). Additional custom physics can be
added via the ``adjust`` member to set or change any loaded data.

.. celerstruct:: inp::FrameworkInput
