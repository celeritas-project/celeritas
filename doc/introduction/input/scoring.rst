.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_scoring:

Scoring
-------

Scoring setup is for measuring and recording physical simulation results.

.. doxygenstruct:: celeritas::inp::Scoring

Geant4 Sensitive detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^

These options are used to integrate Celeritas with Geant4 sensitive detectors
by reconstructing Geant4 hits and calling back to user code.

.. doxygenstruct:: celeritas::inp::GeantSensitiveDetector
.. doxygenstruct:: celeritas::inp::GeantSDStepPointAttributes

Simple calorimeter
^^^^^^^^^^^^^^^^^^

This is used to set up :cpp:class:`celeritas::SimpleCalo`.

.. doxygenstruct:: celeritas::inp::SimpleCalo
