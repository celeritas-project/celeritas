.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_scoring:

Scoring
=======

Scoring setup is for measuring and recording physical simulation results.

.. doxygenstruct:: celeritas::inp::Scoring
   :members:
   :no-link:


Geant4 sensitive detectors
--------------------------

These options are used to integrate Celeritas with Geant4 sensitive detectors
by reconstructing Geant4 hits and calling back to user code.

.. doxygenstruct:: celeritas::inp::GeantSd
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::GeantSdStepPointAttributes
   :members:
   :no-link:


Independent scoring
-------------------

This is used to set up :cpp:class:`celeritas::SimpleCalo`.

.. doxygenstruct:: celeritas::inp::SimpleCalo
   :members:
   :no-link:

