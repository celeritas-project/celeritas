.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _surface:

Surface physics
===============

Optical photons also have special interactions at material boundaries,
specified by user-provided surface properties.
Users can define "boundary" and "interface" surfaces representing,
respectively, the entire boundary of a volume (all points where it touches the
parent or child volumes) and the common face between two adjacent volume
instances.  See :ref:`api_geometry` for a discussion of these definitions and
:ref:`api_geant4_geo` for their translation from Geant4.

.. doxygenclass:: celeritas::optical::VolumeSurfaceSelector

Surface normals are defined by the track position in the geometry. Corrections
may be applied to the geometric surface normal by sampling from a "microfacet
distribution" to account for the roughness of the surface.

.. doxygenclass:: celeritas::optical::SmearRoughnessSampler
.. doxygenclass:: celeritas::optical::GaussianRoughnessSampler

Interactions are sampled from models describing the distributions of absorption,
reflection, and refraction on the surface.

.. doxygenclass:: celeritas::optical::FresnelCalculator
.. doxygenclass:: celeritas::optical::ReflectionFormCalculator
.. doxygenclass:: celeritas::optical::DielectricInteractor
