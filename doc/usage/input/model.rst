.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _model:

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

Surfaces
--------

Surfaces are defined by the relationship between volumes.

.. celerstruct:: inp::Surfaces
.. celerstruct:: inp::Surface
