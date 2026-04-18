.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: console

Plug-ins
========

Celeritas can run as a plugin to different integrated frameworks.

.. _plugins_larsoft:

LArSoft for DUNE
----------------

Celeritas provides interfaces for running optical photon transport within the
LArSoft framework, an integral component of the DUNE software stack.
Celeritas builds the ``PDFullSimCeler`` module to process optical photons from
scintillation.
It requires a ROOT file with ``art::Event`` inputs that have
``sim::SimEnergyDeposit`` object data from the ``IonAndScint`` producer,
exactly as the current ``PDFastSimPAR`` module in LArSoft.
The ``PDFullSimCeler`` module enables replacing the map-based method for
generating the scintillation-to-detector response with full Monte Carlo optical
tracking.

Once Celeritas has been installed (see :ref:`build_ups`), load the
module/library/FHICL paths provided by Celeritas in its install directory (or
build directory if doing development):

.. code::

   $ eval $($CELER_DIR/bin/larceler-env)
   Loaded Celeritas at $CELER_DIR

Then you should be able to include Celeritas components including its photon
detector replacement and analysis modules.

.. literalinclude:: ../../../example/larceler/dune10kt_1x2x6_cpu.fcl
   :language: none
   :start-at: #include

PDFullSimCeler
^^^^^^^^^^^^^^

This "producer" module is a replacement for LArSim's PDFastSimPar using full
optical photon simulation on CPU or GPU.
It tracks photons from simulated energy deposition steps, which are generated
by LArG4 and other modules, and constructs simulated detector hits with
metadata about the tracks that caused the hit.

.. doxygenclass:: celeritas::PDFullSimCeler

Note that this module differs substantially from direct Geant4 integration: it
operates independently from the Geant4 run manager as a "postprocessing" step
in a module after LArG4 runs.

Its configuration options are input via the FHiCL interface:

.. literalinclude:: ../../../src/larceler/PDFullSimCeler.fcl
   :language: none
   :start-after: BEGIN_PROLOG
   :end-before: END_PROLOG

Those options are used during execution time to set up and run a Celeritas
optical physics problem.


GeoSimExporter
^^^^^^^^^^^^^^

This analysis module exports detector geometry data and energy deposition data
for internal testing.

DD4HEP
------

Documentation to be added later.
