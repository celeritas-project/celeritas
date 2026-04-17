.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. **NOTE**: this file is referenced by README.md:
.. if changing the former, update the latter!!

.. highlight:: console

.. _build_ups:

UPS for LArSoft
---------------

Building Celeritas for LArSoft/DUNE is straightforward. First, as described in :ref:`plugins_larsoft`,
you must start up a suitable Apptainer instance for building and execution. Then, load the required LArSoft components.

.. code::

   $ . /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
   Setting up larsoft UPS area... /cvmfs/larsoft.opensciencegrid.org
   Setting up DUNE UPS area... /cvmfs/dune.opensciencegrid.org/products/dune/
   $ setup larsoft v10_14_01 -q e26:prof
   $ setup cmake v3_27_4  || return $?
   $ setup cetmodules v3_24_01 || return $?

.. tip::

   Use the command :samp:`ups list -aK+ {package}` to list available packages.

You can then build and install Celeritas just like any other CMake package.
By default it will autodetect available packages, but Celeritas includes
a preset targeting LArSoft module integration:

.. code::

   $ git clone https://github.com/celeritas-project/celeritas.git
   $ cd celeritas
   $ cmake --preset=larsoft .
   $ cmake --preset=larsoft --install .
