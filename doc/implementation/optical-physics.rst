.. Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_optical_physics:

***************
Optical physics
***************

As with EM physics, the optical physics models in Celeritas are closely related
to those in Geant4. Unlike Geant4, optical photon generation and stepping in
Celeritas takes place in a drastically different manner.

Here is a flowchart depicting the creation of optical photons in the optical
tracking loop:

.. mermaid::

   flowchart TB
     gun["Gun or external"]
     geant4-direct["Direct Geant4 offload"]
     geant4-scint["Geant4 scintillation"]
     geant4-ceren["Geant4 cerenkov"]

     classDef not-impl stroke-width:2px,stroke-dasharray: 5 5
     class geant4-direct,geant4-scint,geant4-ceren not-impl

     subgraph main-celeritas-loop["Main celeritas loop"]
       offload-gather
       scintillation-offload
       cerenkov-offload
     end

     offload-gather -->|pre-step| scintillation-offload
     offload-gather -->|pre-step| cerenkov-offload

     subgraph photon-gen["Optical photon generation"]
       scintillation-gen
       cerenkov-gen
     end

     scintillation-offload -->|generator dist| scintillation-gen
     cerenkov-offload -->|generator dist| cerenkov-gen
     geant4-scint -->|generator dist| scintillation-gen
     geant4-ceren -->|generator dist| cerenkov-gen


     photons["Optical tracking loop"]
     gun -->|primaries| photons

     geant4-direct -->|primaries| photons
     scintillation-gen -->|primaries| photons
     cerenkov-gen -->|primaries| photons


Optical materials
=================

Each "physics material" (see :cpp:class:`celeritas::ImportPhysMaterial`) can
have an associated "optical material." When importing from Geant4, each optical
material corresponds to a single "geometry material" (see
:cpp:class:`celeritas::ImportGeoMaterial`) that has a ``RINDEX`` material
property, and all physical materials that use the geometry material share
the same optical material.

.. doxygenclass:: celeritas::optical::MaterialParams

Offloading
==========

During the main :ref:`stepping loop <api_stepping>`, the :cpp:class:`celeritas::OpticalCollector`
class adds a pre-step hook to store each track's speed, position, time, and
material; at the end of the step, the track's updated properties and
within-step energy distribution are used to "offload" optical photons by
generating *distribution parameters* to be sampled in the stepping loop.

.. doxygenclass:: celeritas::OpticalCollector
.. doxygenclass:: celeritas::CerenkovOffload
.. doxygenclass:: celeritas::ScintillationOffload
.. doxygenstruct:: celeritas::optical::GeneratorDistributionData

Generating
==========

Depending on the process that emitted a photon, the "generator" classes
sample from the distribution of photons specified by the
"generator distribution" to create optical photon *primaries*.

.. doxygenclass:: celeritas::optical::CerenkovGenerator
.. doxygenclass:: celeritas::optical::ScintillationGenerator

Volumetric processes
====================

Like other particles, optical photons undergo stochastic interactions inside
optical materials.

.. todo:: Add this section once rayleigh scattering, absorption, etc. are
   completed.

Surface processes
=================

Optical photons also have special interactions at material boundaries. These
boundaries are imported from Geant4 using the "skin" definitions that specify
properties of a volume's outer surface or of the surface between two specific
volumes.

.. todo:: Add this section once surface models are implemented.

Imported data
=============

In addition to the core :ref:`api_importdata`, these import parameters are used
to provide cross sections, setup options, and other data to the optical physics.

.. doxygenstruct:: celeritas::ImportOpticalAbsorption
.. doxygenstruct:: celeritas::ImportOpticalMaterial
.. doxygenstruct:: celeritas::ImportOpticalParameters
.. doxygenstruct:: celeritas::ImportOpticalProperty
.. doxygenstruct:: celeritas::ImportOpticalRayleigh

.. doxygenstruct:: celeritas::ImportScintComponent
.. doxygenstruct:: celeritas::ImportScintData
.. doxygenstruct:: celeritas::ImportParticleScintSpectrum
.. doxygenstruct:: celeritas::ImportMaterialScintSpectrum

.. doxygenstruct:: celeritas::ImportWavelengthShift

