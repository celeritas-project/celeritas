.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_standalone_input:

Standalone execution
====================

The ``StandaloneInput`` and ``OpticalStandaloneInput`` provide the complete
configuration required to run a standalone Celeritas simulation, for example,
using the ``celer-sim`` and ``celer-optical`` applications. They specify the
system and problem configuration, how the physics data is loaded, and the
events that are run.

.. celerstruct:: inp::StandaloneInput
.. celerstruct:: inp::OpticalStandaloneInput
