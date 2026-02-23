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

TODO: mapping of Geant4 model+finish to Celeritas surface order types
