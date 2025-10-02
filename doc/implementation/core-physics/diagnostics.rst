.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_diagnostics:

Diagnostics
===========

Several classes can add kernels to the stepping loop to extract data about the
tracks in flight, on CPU or GPU.  The interface to most of these diagnostics
should be considered unstable.

.. doxygenclass:: celeritas::ActionDiagnostic

.. doxygenclass:: celeritas::StepDiagnostic

.. doxygenclass:: celeritas::SlotDiagnostic


Step writers
------------

These use a special interface to extract step information.

.. doxygenclass:: celeritas::SimpleCalo

.. doxygenclass:: celeritas::RootStepWriter

.. celerstruct:: SimpleRootFilterInput
