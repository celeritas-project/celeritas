.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _model:

Model
=====

This specifies the problem geometry and material properties. See
:ref:`api_geometry` for details on the definition of volumes and surfaces.

.. doxygenstruct:: celeritas::inp::Model
   :members:
   :no-link:

Volumes
-------

These input classes describe the volume hierarchy.

.. doxygenstruct:: celeritas::inp::Volumes
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::Volume
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::VolumeInstance
   :members:
   :no-link:

Surfaces
--------

Surfaces are defined by the relationship between volumes.

.. doxygenstruct:: celeritas::inp::Surfaces
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::Surface
   :members:
   :no-link:
