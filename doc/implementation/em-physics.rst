.. Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_em_physics:

**********
EM Physics
**********

The physics models in Celeritas are primarily derived from references cited by
Geant4, including the Geant4 physics reference manual. Undocumented adjustments
to those models in Geant4 may also be implemented, and hopefully, explained in
our documentation.

Distributions
=============

At a higher level, Celeritas expresses many physics operations as
distributions of *updated* track states based on *original* track states. For
example, the Tsai-Urban distribution used for sampling exiting angles of
bremsstrahlung and pair production has parameters of incident particle energy
and mass, and it samples the exiting polar angle cosine.

.. doxygenclass:: celeritas::BhabhaEnergyDistribution

.. doxygenclass:: celeritas::EnergyLossGammaDistribution

.. doxygenclass:: celeritas::EnergyLossGaussianDistribution

.. doxygenclass:: celeritas::EnergyLossUrbanDistribution

.. doxygenclass:: celeritas::MollerEnergyDistribution

.. doxygenclass:: celeritas::TsaiUrbanDistribution


Processes and models
====================

Additional distributions are built on top of the helper distributions above.
All discrete interactions (in Geant4 parlance, "post-step do-it"s) use
distributions to sample an *Interaction* based on incident particle properties.
The sampled result contains the updated particle direction and energy, as well
as properties of any secondary particles produced.

Ionization
----------

.. doxygenclass:: celeritas::BraggICRU73QOInteractor
.. doxygenclass:: celeritas::MollerBhabhaInteractor
.. doxygenclass:: celeritas::MuBetheBlochInteractor

Bremsstrahlung
--------------

.. doxygenclass:: celeritas::RelativisticBremInteractor
.. doxygenclass:: celeritas::SeltzerBergerInteractor
.. doxygenclass:: celeritas::MuBremsstrahlungInteractor


The Seltzer--Berger interactions are sampled with the help of an energy
distribution and cross section correction:

.. doxygenclass:: celeritas::SBEnergyDistribution
.. doxygenclass:: celeritas::detail::SBPositronXsCorrector


Scattering
----------

.. doxygenclass:: celeritas::CoulombScatteringInteractor
.. doxygenclass:: celeritas::KleinNishinaInteractor
.. doxygenclass:: celeritas::RayleighInteractor

Conversion/annihilation/photoelectric
-------------------------------------

.. doxygenclass:: celeritas::BetheHeitlerInteractor
.. doxygenclass:: celeritas::EPlusGGInteractor
.. doxygenclass:: celeritas::LivermorePEInteractor

.. doxygenclass:: celeritas::AtomicRelaxation

Multiple scattering
-------------------

.. doxygenclass:: celeritas::detail::UrbanMscSafetyStepLimit
.. doxygenclass:: celeritas::detail::UrbanMscScatter

Continuous slowing down
=======================

Most charged interactions emit one or more low-energy particles during their
interaction. Instead of creating explicit daughter tracks that are
immediately killed due to low energy, part of the interaction cross section is
lumped into a "slowing down" term that continuously deposits energy locally
over the step. This mean energy loss term is an approximation; additional
models are implemented to adjust the loss per step with stochastic sampling for
improved accuracy.

.. doxygenclass:: celeritas::EnergyLossHelper

Imported data
=============

In addition to the core :ref:`api_importdata`, these import parameters are used
to provide cross sections, setup options, and other data to the EM physics.

.. doxygenstruct:: celeritas::ImportEmParameters
.. doxygenstruct:: celeritas::ImportAtomicTransition
.. doxygenstruct:: celeritas::ImportAtomicSubshell
.. doxygenstruct:: celeritas::ImportAtomicRelaxation

.. doxygenstruct:: celeritas::ImportLivermoreSubshell
.. doxygenstruct:: celeritas::ImportLivermorePE

.. doxygenstruct:: celeritas::ImportSBTable
