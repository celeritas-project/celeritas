.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_importdata:

Imported data
=============

.. note:: These classes will be merged into the problem input definition
   :ref:`input`.

Celeritas reads physics data from Geant4 (or from a ROOT file exported from
data previously loaded into Geant4). Different versions of Geant4 (and Geant4
data) can be used seamlessly with any version of Celeritas, allowing
differences to be isolated without respect to machine or model implementation.
The following classes enumerate the core data loaded at runtime.

.. doxygenstruct:: celeritas::ImportData
   :members:
   :undoc-members:
   :no-link:

Material and geometry properties
--------------------------------

.. doxygenstruct:: celeritas::ImportIsotope
.. doxygenstruct:: celeritas::ImportElement
.. doxygenstruct:: celeritas::ImportMatElemComponent
.. doxygenstruct:: celeritas::ImportGeoMaterial
.. doxygenstruct:: celeritas::ImportProductionCut
.. doxygenstruct:: celeritas::ImportPhysMaterial
.. doxygenstruct:: celeritas::ImportRegion
.. doxygenstruct:: celeritas::ImportVolume
.. doxygenstruct:: celeritas::ImportTransParameters
.. doxygenstruct:: celeritas::ImportLoopingThreshold

.. doxygenenum:: celeritas::ImportMaterialState
   :no-link:

Physics properties
------------------

.. doxygenstruct:: celeritas::ImportParticle
.. doxygenstruct:: celeritas::ImportProcess
.. doxygenstruct:: celeritas::ImportModel
.. doxygenstruct:: celeritas::ImportMscModel
.. doxygenstruct:: celeritas::ImportModelMaterial
.. doxygenstruct:: celeritas::ImportPhysicsTable
.. doxygenstruct:: celeritas::ImportPhysicsVector

.. doxygenenum:: celeritas::ImportUnits
   :no-link:
