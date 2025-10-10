.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_model:

Model
=====

This specifies the problem geometry and material properties. See
:ref:`api_geometry` for details on the definition of volumes and surfaces.

.. celerstruct:: inp::Model

Volumes
-------

These input classes describe the volume hierarchy.

.. celerstruct:: inp::Volumes
.. celerstruct:: inp::Volume
.. celerstruct:: inp::VolumeInstance

.. _inp_surfaces:

Surfaces
--------

Surfaces are defined by the relationship between volumes. These definitions are
loaded from Geant4's "skin" and "border" surface tables into
Celeritas "boundary" and "interface" surfaces, respectively (see :ref:`api_geant4_geo`).

.. celerstruct:: inp::Surfaces
.. celerstruct:: inp::Surface

Regions
-------

Regions are groups of volumes that share configurable properties or behaviors.
Although not yet used in Celeritas (see `issue #983`_), these are loaded
from Geant4 to provide configurable physics fidelity and selection for
performance optimization.

.. _issue #983: https://github.com/celeritas-project/celeritas/issues/983

.. celerstruct:: inp::Regions
.. celerstruct:: inp::Region

Detectors
---------

Detector information is gathered from Geant4 to map volumes to user sensitive
detectors.

.. celerstruct:: inp::Detectors
.. celerstruct:: inp::Detector


Materials
---------

Fundamental material, elemental, and isotopic properties will be added to the
model.
