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
   $ setup larsoft v10_14_01 -q e26:prof
   $ setup -B dunesw v10_14_01d00 -q e26:prof

If running

.. sourcecode::

   $ git clone https://github.com/celeritas-project/celeritas.git
   Cloning into 'celeritas'...
   # ...

   $ cmake --preset=default . -DCELERITAS_USE_LArSoft=ON
   # ...
   -- Build files have been written to: /scratch/sethj/larsoft-dev/celeritas/build-default

   $ cmake --preset=default . -DCELERITAS_USE_LArSoft=ON  -DCMAKE_INSTALL_PREFIX=$PWD/install
   # ...
   -- Build files have been written to: /scratch/sethj/larsoft-dev/celeritas/build-default
   $ cd build-default/ && ninja install


DD4HEP
------

Documentation to be added later.
