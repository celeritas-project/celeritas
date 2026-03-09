.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _surface:

Surface physics
===============

Optical photons have special interactions at material boundaries,
specified by user-provided surface properties or the adjacent bulk material
properties.

.. doxygenclass:: celeritas::optical::SurfacePhysicsParams

.. doxygenenum:: celeritas::optical::SurfacePhysicsOrder

During surface crossing, tracks maintain a "within-surface state" that stores
the current layer and direction, decoupling the geometry state from the
material state.

.. doxygenclass:: celeritas::optical::SurfaceTraversalView

.. _surface_boundary_init:

Boundary initialization
-----------------------

Users can define "boundary" and "interface" surfaces representing,
respectively, the entire boundary of a volume (all points where it touches the
parent or child volumes) and the common face between two adjacent volume
instances.  See :ref:`api_geometry` for a discussion of these definitions and
:ref:`api_geant4_geo` for their translation from Geant4.

.. doxygenclass:: celeritas::optical::VolumeSurfaceSelector

.. _surface_roughness:

Roughness
---------

Surface normals are defined by the track position in the geometry. Corrections
may be applied to the geometric (macroscopic) surface normal by sampling from a
"microfacet distribution" to account for the roughness of the surface.

.. doxygenclass:: celeritas::optical::SmearRoughnessSampler
.. doxygenclass:: celeritas::optical::GaussianRoughnessSampler

These samplers may use rejection based on the macroscopic surface normal.

.. doxygenclass:: celeritas::optical::EnteringSurfaceNormalSampler

.. _surface_reflectivity:

Reflectivity
------------

To support custom overrides of standard surface physics, Geant4 allows surfaces
to have user-defined grids which determine the frequency at which optical
photons should bypass the surface. A transmittance value greater than zero
allows photons to be transmitted across the surface without change, bypassing
all surface interactions. A reflectivity value below one allows photons to be
absorbed directly on the surface.

While Celeritas supports these grid-based overrides for Geant4 compatibility,
similar effects for most use cases may be achieved by using Celeritas-specific
roughness and interaction models without the need to specify dummy values.

.. doxygenclass:: celeritas::optical::GridReflectivityModel
.. doxygenclass:: celeritas::optical::FresnelReflectivityModel

.. _surface_interaction:

Interaction
-----------

Interactions are sampled from models describing the distributions of absorption,
reflection, and refraction on the surface.

.. doxygenclass:: celeritas::optical::FresnelCalculator
.. doxygenclass:: celeritas::optical::ReflectionFormCalculator
.. doxygenclass:: celeritas::optical::DielectricInteractor
.. doxygenstruct:: celeritas::optical::OnlyReflectionExecutor
.. doxygenclass:: celeritas::optical::TrivialInteractor

.. _surface_boundary_post:

Boundary crossing
-----------------

If a track exits the surface (i.e., in the initial and final layer, heading
away from the surface), the geometry state is updated accordingly at the end of
the loop.
