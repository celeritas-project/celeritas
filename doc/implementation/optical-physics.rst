.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_optical_physics:

***************
Optical physics
***************

As with EM physics, the optical physics models in Celeritas are closely related
to those in Geant4. Unlike Geant4, optical photon generation and stepping in
Celeritas takes place in a drastically different manner.
:numref:`fig-optical-flow` depicts the creation of optical photons in the
optical tracking loop as a flowchart.

.. _fig-optical-flow:

.. figure:: /_static/mermaid/optical-flow.*
   :align: center
   :width: 80%

   Creation of optical photons in the optical tracking loop.


Optical materials
=================

Each "physics material" (i.e., a combination of material and physics options) can
have an associated "optical material" (compatible with optical photons).

.. doxygenclass:: celeritas::optical::MaterialParams

Geant4 integration
------------------

When importing from Geant4, each optical material corresponds to a single
:cpp:class:`G4MaterialPropertiesTable` that has a ``RINDEX`` material property.
(It also provides a special case for water if no material table is associated,
allowing Rayleigh scattering by default by providing an isothermal
compressibility and assuming STP.)

Celeritas translates many Geant4 material properties into its internal physics
input parameters. It also allows material-specific user configuration of
Celeritas-only physics, using properties listed in the following table.

.. table:: Celeritas-only properties, with the ``CELER_`` prefix omitted.

   +-------------------------------------+-------------------------------------------------------------+
   | Name                                | Description                                                 |
   +=====================================+=============================================================+
   | :code:`SCINTILLATIONLAMBDAMEAN`     | Mean wavelength of the Gaussian scintillation peak [mm]     |
   +-------------------------------------+-------------------------------------------------------------+
   | :code:`SCINTILLATIONLAMBDASIGMA`    | Standard deviation of the Gaussian scintillation peak [mm]  |
   +-------------------------------------+-------------------------------------------------------------+


Offloading
==========

During the main :ref:`stepping loop <api_stepping>`, the :cpp:class:`celeritas::OpticalCollector`
class adds a pre-step hook to store each track's speed, position, time, and
material; at the end of the step, the track's updated properties and
within-step energy distribution are used to "offload" optical photons by
generating *distribution parameters* to be sampled in the stepping loop. The
*generator distribution* data is analogous to the "genstep" data structure in
Opticks :cite:`blyth-opticks-2019`.

.. doxygenclass:: celeritas::OpticalCollector
.. doxygenclass:: celeritas::CherenkovOffload
.. doxygenclass:: celeritas::ScintillationOffload
.. celerstruct:: optical::GeneratorDistributionData

Generating
==========

Depending on the process that emitted a photon, the generator classes
sample from the distribution of photons specified by the
generator distribution to create optical photon *initializers* which are
analogous to secondary particles in Geant4.

.. doxygenclass:: celeritas::optical::CherenkovGenerator
.. doxygenclass:: celeritas::optical::ScintillationGenerator

Volumetric processes
====================

Like other particles, optical photons undergo stochastic interactions inside
optical materials.

.. doxygenclass:: celeritas::optical::AbsorptionModel
.. doxygenclass:: celeritas::optical::RayleighModel
.. doxygenclass:: celeritas::optical::RayleighMfpCalculator

.. _surface_processes:

Surface processes
=================

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
