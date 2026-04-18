.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: console

Plug-ins
========

Celeritas can run as a plugin to different integrated frameworks.

.. _plugins_larsoft:

LArSoft for DUNE
----------------

LArSoft is an integral component of the DUNE simulation framework.
Celeritas builds the ``PDFullSimCeler`` module to process optical photons from
scintillation.
It requires ROOT input file with ``art::Event``
``sim::SimEnergyDeposit``object data from the ``IonAndScint`` producer, exactly
as the current ``PDFastSimPAR`` module in LArSoft.
The ``PDFullSimCeler`` module enables replacing the map-based method for
generating the scintillation-to-detector response by a full Monte Carlo optical
tracking.

Once Celeritas has been installed (see :ref:`build_ups`), load the
module/library/FHICL paths provided by Celeritas in its install directory (or
build directory if doing development):

.. code::

   $ eval $($CELER_DIR/bin/larceler-env)
   Loaded Celeritas at $CELER_DIR

Then you should be able to include Celeritas components including its photon
detector replacement and analysis modules.

.. literalinclude:: ../../example/larceler/dune10kt_1x2x6_cpu.fcl
   :language: none
   :start-at: #include

PDFullSimCeler
""""""""""""""

This "producer" module is a replacement for LArSim's PDFastSimPar.

GeoSimExporter
""""""""""""""

This analysis module exports detector geometry data and energy deposition data
for internal testing.

DD4HEP
------

Documentation to be added later.
