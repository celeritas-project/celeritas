.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Celeritas setup
===============

The setup options help translate the Geant4 physics and problem setup to
Celeritas. They are also necessary to set up the GPU offloading
characteristics. Future versions of Celeritas will automate more of these
settings.

By default, sensitive detectors are automatically mapped from Geant4 to
Celeritas using the ``enabled`` option of
:cpp:struct:`celeritas::SDSetupOptions`. If no SDs are present (e.g., in a test
problem, or one which has only a "stepping manager" which is not presently
compatible with Celeritas), the Celeritas setup will fail with an error like:

.. code-block:: none

   *** G4Exception : celer0001
         issued by : accel/detail/GeantSd.cc:210
   Celeritas runtime error: no G4 sensitive detectors are defined: set `SetupOptions.sd.enabled` to `false` if this is expected
   *** Fatal Exception *** core dump ***


.. celerstruct:: SetupOptions
.. celerstruct:: SDSetupOptions

Helper functions
----------------

This function helps find logical volumes from volume names:

.. doxygenfunction:: celeritas::FindVolumes

Magnetic field setup
--------------------

The magnetic fields defined for Celeritas can be used as Geant4 native magnetic
fields (see :ref:`api_accel_adapters`).

.. doxygenclass:: celeritas::UniformAlongStepFactory
.. doxygenclass:: celeritas::RZMapFieldAlongStepFactory
.. doxygenclass:: celeritas::CylMapFieldAlongStepFactory

.. _g4_ui_macros:

Macro UI Options
----------------

The :cpp:class:`celeritas::SetupOptionsMessenger`, instantiated automatically
by the Integration helper classes, provides a Geant4 "UI" macro interface to
many of the options.

.. doxygenclass:: celeritas::SetupOptionsMessenger
