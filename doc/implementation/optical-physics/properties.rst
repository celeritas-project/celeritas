.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _optical_properties:

Optical properties
==================

Each "physics material" (i.e., a combination of material and physics options) can
have an associated "optical material" (compatible with optical photons).

.. doxygenclass:: celeritas::optical::MaterialParams

Users can define "boundary" and "interface" surfaces representing,
respectively, the entire boundary of a volume (all points where it touches the
parent or child volumes) and the common face between two adjacent volume
instances.  See :ref:`api_geometry` for a discussion of these definitions and
:ref:`api_geant4_geo` for their translation from Geant4.

Bulk properties
---------------

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

Surface properties
------------------

Geant4 surfaces may be configured with the following variables:

.. table:: Geant4 surface parameters.

    +--------+------------------------------------------------------------------+
    | Name   | Description                                                      |
    +========+==================================================================+
    | model  | Model used for optical surface physics                           |
    +--------+------------------------------------------------------------------+
    | finish | Level of a surface's roughness                                   |
    +--------+------------------------------------------------------------------+
    | type   | Type of optical materials on either side of the surface          |
    +--------+------------------------------------------------------------------+
    | value  | Used by the surface finish to parameterize the surface roughness |
    +--------+------------------------------------------------------------------+

.. table:: Model independent surface properties.

    +-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                  | Description                                                                                                                               |
    +=======================+===========================================================================================================================================+
    | :code:`REFLECTIVITY`  | One minus the probability the photon is absorbed on the surface.                                                                          |
    +-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
    | :code:`TRANSMITTANCE` | Probability the photon is transmitted across the surface without change.                                                                  |
    +-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
    | :code:`EFFICIENCY`    | Quantum efficiency of the detector attached to the surface. If a photon is absorbed, this is the probability it is subsequently detected. |
    +-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------+

.. _optical_glisur_model:

GLISUR Model
^^^^^^^^^^^^

Geant3's default model, referred to as ``GLISUR``, supports a uniform smear
roughness for both dielectric-dielectric and dieletric-metal surfaces.

.. table:: Mapping of GLISUR model finishes to Celeritas inputs (in the ``celeritas::inp`` namespace).

    +--------------+----------------------------------------------+-------------------------------------+
    | Finish       | Roughness                                    | Reflection Mode                     |
    +==============+==============================================+=====================================+
    | ``polished`` | No roughness                                 | :code:`ReflectionForm::from_spike`  |
    +--------------+----------------------------------------------+-------------------------------------+
    | ``ground``   | Smear roughness with roughness ``1 - value`` | :code:`ReflectionForm::from_lobe`   |
    +--------------+----------------------------------------------+-------------------------------------+

.. table:: Mapping of GLISUR model types to Celeritas inputs (in the ``celeritas::inp`` namespace).

   +---------------------------+------------------------------------------------+
   | Type                      | Interaction                                    |
   +===========================+================================================+
   | ``dielectric_dielectric`` | :code:`DielectricInteraction::from_dielectric` |
   +---------------------------+------------------------------------------------+
   | ``dielectric_metal``      | :code:`DielectricInteraction::from_metal`      |
   +---------------------------+------------------------------------------------+

.. _optical_unified_model:

UNIFIED Model
^^^^^^^^^^^^^

The UNIFIED model provides a probabilistic set of possible types of reflection
a photon may undergo. These are stores as properties on a surface's property
table, which is loaded into Celeritas as a
:code:`celeritas::inp::ReflectionForm`

.. table:: Description of UNIFIED reflection modes.

    +-------------------------------+--------------------------------------------------------------------------------------------+
    | Name                          | Description                                                                                |
    +===============================+============================================================================================+
    | :code:`SPECULARSPIKECONSTANT` | Probability to reflect with respect to the surface's global normal (no roughness sampling) |
    +-------------------------------+--------------------------------------------------------------------------------------------+
    | :code:`SPECULARLOBECONSTANT`  | Probability to reflect with respect to a facet normal as sampled by the surface roughness  |
    +-------------------------------+--------------------------------------------------------------------------------------------+
    | :code:`BACKSCATTERCONSTANT`   | Probability to directly back-scatter (opposite direction and polarization)                 |
    +-------------------------------+--------------------------------------------------------------------------------------------+

As with the ``GLISUR`` model, the surface type determines the Celeritas
interaction model:

.. table:: Mapping of UNIFIED model types to Celeritas inputs (in the ``celeritas::inp`` namespace).

   +---------------------------+------------------------------------------------+
   | Type                      | Interaction                                    |
   +===========================+================================================+
   | ``dielectric_dielectric`` | :code:`DielectricInteraction::from_dielectric` |
   +---------------------------+------------------------------------------------+
   | ``dielectric_metal``      | :code:`DielectricInteraction::from_metal`      |
   +---------------------------+------------------------------------------------+

Both of which support the ``polished`` and ``ground`` finishes as well:

.. table:: Mapping of UNIFIED model basic finishes to Celeritas inputs (in the ``celeritas::inp`` namespace).

    +--------------+-------------------------------------------------+-----------------------------------------+
    | Finish       | Roughness                                       | Reflection Mode                         |
    +==============+=================================================+=========================================+
    | ``polished`` | No roughness                                    | :code:`ReflectionForm::from_spike`      |
    +--------------+-------------------------------------------------+-----------------------------------------+
    | ``ground``   | Gaussian roughness with ``sigma_alpha = value`` | Populated from UNIFIED reflection modes |
    +--------------+-------------------------------------------------+-----------------------------------------+

The UNIFIED model further supports front-painted and back-painted finishes for
dielectric-dielectric interfaces. A painted surface is entirely reflective with
a fixed reflection mode.

.. table:: Mapping of UNIFIED model dielectric-dielectric front-painted finishes to Celeritas inputs (in the ``celeritas::inp`` namespace).

   +--------------------------+--------------+----------------------------------------+
   | Finish                   | Roughness    | Interaction                            |
   +==========================+==============+========================================+
   | ``polishedfrontpainted`` | No roughness | :code:`ReflectionMode::specular_spike` |
   +--------------------------+--------------+----------------------------------------+
   | ``groundfrontpainted``   | No roughness | :code:`ReflectionMode::diffuse_lobe`   |
   +--------------------------+--------------+----------------------------------------+

Back-painted finishes are used to represent surfaces such as a "crystal - air
gap - wrapping" where there's an interstitial material between the initial
material and the wrapping. The ``RINDEX`` present on a surface material
property table represents the refractive index of the interstitial material when
applying the Fresnel equations between the initial and interstitial materials.
Furthermore, if reflectivity grids are defined on the surface, they only apply
to the initial-interstitial interface and not the painted surface for
back-painted finishes.

Back-painted surfaces are represented as combinations of the above surfaces,
with the interstitial material determined by the ``RINDEX`` of the surface
material property table.

.. table:: Surface construction of UNIFIED model dielectric-dielectric back-painted finishes in Celeritas.

   +-------------------------+--------------------------------------+----------------------------+
   | Finish                  | Initial-Interstitial Surface         | Interstitial-Final Surface |
   +=========================+======================================+============================+
   | ``polishedbackpainted`` | UNIFIED dielectric-dielectric ground | Polished front-painted     |
   +-------------------------+--------------------------------------+----------------------------+
   | ``groundbackpainted``   | UNIFIED dielectric-dielectric ground | Ground front-painted       |
   +-------------------------+--------------------------------------+----------------------------+
