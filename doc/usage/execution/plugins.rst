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

.. sourcecode::

   $ mkdir larsoft-dev
   $ cd larsoft-dev/
   $ mrb newDev

   building development area for larsoft v10_14_01 -q e26:prof


   The following configuration is defined:
     The top level directory is .
     The source code directory will be under .
     The build directory will be under .
     The local product directory will be under .

   MRB_BUILDDIR is $SCRATCH/larsoft-dev/build_slf7.x86_64
   MRB_SOURCE is $SCRATCH/larsoft-dev/srcs
   INFO: copying /cvmfs/larsoft.opensciencegrid.org/products/larsoft/v10_14_01/releaseDB/base_dependency_database

   IMPORTANT: You must type
       source $SCRATCH/larsoft-dev/localProducts_larsoft_v10_14_01_e26_prof/setup
   NOW and whenever you log in

   $ source $SCRATCH/larsoft-dev/localProducts_larsoft_v10_14_01_e26_prof/setup

   MRB_PROJECT=larsoft
   MRB_PROJECT_VERSION=v10_14_01
   MRB_QUALS=e26:prof
   MRB_TOP=$SCRATCH/larsoft-dev
   MRB_SOURCE=$SCRATCH/larsoft-dev/srcs
   MRB_BUILDDIR=$SCRATCH/larsoft-dev/build_slf7.x86_64
   MRB_INSTALL=$SCRATCH/larsoft-dev/localProducts_larsoft_v10_14_01_e26_prof

   PRODUCTS=$SCRATCH/larsoft-dev/localProducts_larsoft_v10_14_01_e26_prof:/cvmfs/dune.opensciencegrid.org/products/dune:/cvmfs/larsoft.opensciencegrid.org/products:/cvmfs/larsoft.opensciencegrid.org/packages:/cvmfs/fermilab.opensciencegrid.org/products/common/db/
   CETPKG_INSTALL=$SCRATCH/larsoft-dev/localProducts_larsoft_v10_14_01_e26_prof

Clone and install the fork of larsim:

.. sourcecode::

   $ mrb g https://github.com/nuRiceLab/larsim.git
   Cloning into 'larsim'...
   # ...
   NOTICE: Adding larsim to CMakeLists.txt file
   $ mrbsetenv
   The working build directory is $SCRATCH/larsoft-dev/build_slf7.x86_64
   The source code directory is $SCRATCH/larsoft-dev/srcs
   ----------- check this block for errors -----------------------
   ----------------------------------------------------------------
   To inspect build variable settings, execute $SCRATCH/larsoft-dev/build_slf7.x86_64/cetpkg_info.sh

   Please use "buildtool" (or "mrb b") to configure and build MRB project "larsoft", e.g.:

     buildtool -vTl [-jN]

   See "buildtool --usage" (short usage help) or "buildtool -h|--help"
   (full help) for more details.

   $ mrb i -G Ninja
   INFO: install prefix = $SCRATCH/larsoft-dev/localProducts_larsoft_v10_14_01_e26_prof
   # ...

   ------------------------------------
   INFO: stage install SUCCESS for MRB project larsoft v10_14_01
   ------------------------------------

Finally, build and install Celeritas (you can also run directly from the build
directory):

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
