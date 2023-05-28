.. Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _example_geant_full:

Geant4 integration
==================

This lengthy example is a generic demonstration of using Celeritas to offload
EM tracks from a Geant4 application. It is compatible with Geant4 10.7--11.0.
See :ref:`accel` for API documentation of the Celeritas classes being called.

Main executable
---------------

The main method creates the action initialization and reads user setup options
through a macro file. The setup options, aside from the detector geometry file
and event input file, map to options in the :class:`celeritas::SetupOptions`
class.

.. literalinclude:: ../../app/celer-g4/celer-g4.cc
   :start-at: #include

.. literalinclude:: ../../app/celer-g4/GlobalSetup.hh
   :start-at: #pragma

User initialization
-------------------

.. literalinclude:: ../../app/celer-g4/DetectorConstruction.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/DetectorConstruction.cc
   :start-at: #include

.. literalinclude:: ../../app/celer-g4/ActionInitialization.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/ActionInitialization.cc
   :start-at: #include

User actions
------------

The user actions are responsible for setting up Celeritas (using the
Acceleritas interface) and passing tracks to be offloaded.

.. literalinclude:: ../../app/celer-g4/PrimaryGeneratorAction.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/RunAction.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/RunAction.cc
   :start-at: #include

.. literalinclude:: ../../app/celer-g4/EventAction.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/EventAction.cc
   :start-at: #include

.. literalinclude:: ../../app/celer-g4/TrackingAction.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/TrackingAction.cc
   :start-at: #include

Sensitive Detectors
-------------------

The SD setup options in the ``DetectorConstruction`` constructor should be set
as the union of requirements over all user Geant4 sensitive detectors.  The
sensitive detector here is pretty generic and requires only the touchable, the
energy deposition, and the time.

.. literalinclude:: ../../app/celer-g4/SensitiveDetector.hh
   :start-at: #pragma

.. literalinclude:: ../../app/celer-g4/SensitiveDetector.cc
   :start-at: #include

.. literalinclude:: ../../app/celer-g4/SensitiveHit.hh
   :start-at: #pragma

