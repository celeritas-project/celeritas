.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_import:

Loading data into Celeritas
===========================

Problem data can be imported directly from Geant4 or loaded from a ROOT file.

.. doxygentypedef:: celeritas::inp::PhysicsImport

The following input types select the source of the imported physics data.

.. celerstruct:: inp::PhysicsFromFile
.. celerstruct:: inp::PhysicsFromGeant
.. celerstruct:: inp::PhysicsFromGeantFiles
