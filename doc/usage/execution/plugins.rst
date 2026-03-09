.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: console

Plug-ins
========

Celeritas can run as a plugin to different integrated frameworks.

.. _plugins_larsoft:

LArSoft for DUNE
----------------

LArSoft is an integral component of the DUNE simulation framework. Celeritas
builds a tool to process optical photons from scintillation. It requires a
soon-to-be-merged fork_ of LArSoft that refactors the scintillation-to-detector
response calculation to allow Monte Carlo optical tracking as an alternative to
the current map-based method.

.. _fork: https://github.com/nuRiceLab/larsim

Building Celeritas as a LArSoft extension requires the whole larsoft toolchain,
available on Fermilab's ``scisoftbuild01``. The environment script at
``env/scisoftbuild01.sh`` can be sourced at startup to define an
``apptatiner_fermilab`` function that launches the container needed to build
and run.

Once inside the apptainer, initialize the UPS packaging system and load LArSoft and DUNE
components:

.. sourcecode::

   $ . /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
   Setting up larsoft UPS area... /cvmfs/larsoft.opensciencegrid.org
   Setting up DUNE UPS area... /cvmfs/dune.opensciencegrid.org/products/dune/
   $ setup -B dunesw v10_14_01d00 -q e26:prof

Finally, load the module/library/FHICL paths provided by Celeritas:

.. sourcecode::

   $ eval $($SCRATCHDIR/build/celeritas-reldeb-orange/bin/larceler-env)
   Loaded Celeritas at .../build/celeritas-reldeb-orange

Then you should be able to include Celeritas components.

.. sourcecode:: none

   #include "PDFullSimCeler.fcl"
   #include "standard_g4_dune10kt_1x2x6.fcl"

   dunefd_pdfullsim_cpu: {
     @table::PDFullSimCeler
   }

   # Use Celeritas full sim configuration
   physics.producers.PDFastSim: @local::dunefd_pdfullsim_cpu

DD4HEP
------

Documentation to be added later.
