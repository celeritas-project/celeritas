.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: console

Plug-ins
========

Celeritas can run as a plugin to different integrated frameworks.

LArSoft
-------

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

Once inside the apptainer, initialize the UPS packaging system and load LArSoft
components:

.. sourcecode::

   $ . /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
   Setting up larsoft UPS area... /cvmfs/larsoft.opensciencegrid.org
   Setting up DUNE UPS area... /cvmfs/dune.opensciencegrid.org/products/dune/
   $ setup larsoft v10_14_01 -q e26:prof

Then create a local build area with Fermilab's MRB tool:

Finally, install Celeritas:



DD4HEP
------

Documentation to be added later.
