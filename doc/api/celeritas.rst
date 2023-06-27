.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_celeritas:

Celeritas
=========

The ``celeritas`` directory focuses on the physics and transport loop
implementation for the Celeritas codebase, using components from the
``corecel`` and ``orange`` dependencies.

Fundamentals
------------

.. _api_units:

.. doxygennamespace:: units

.. _api_constants:

.. doxygennamespace:: constants

.. doxygenfile:: celeritas/Units.hh
   :sections: user-defined var innernamespace

.. doxygenfile:: celeritas/Quantities.hh
   :sections: user-defined var innernamespace

.. doxygenfile:: celeritas/Constants.hh
   :sections: user-defined var innernamespace

Problem definition
------------------

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams

Transport interface
-------------------

.. doxygenclass:: celeritas::Stepper

Geant4 physics interfaces
-------------------------

.. doxygenclass:: celeritas::GeantImporter

.. doxygenclass:: celeritas::GeantSetup

.. _api_geant4_physics_options:

Geant4 physics options
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::GeantPhysicsOptions

Geometry interfaces
-------------------

.. doxygenclass:: celeritas::VecgeomParams

.. doxygenclass:: celeritas::GeantGeoParams

On-device access
----------------

.. doxygenclass:: celeritas::MaterialTrackView

.. doxygenclass:: celeritas::ParticleTrackView

.. doxygenclass:: celeritas::PhysicsTrackView


.. _celeritas_random:

Random number distributions
---------------------------

The 2011 ISO C++ standard defined a new functional paradigm for sampling from
random number distributions. In this paradigm, random number *engines* generate
a uniformly distributed stream of bits. Then, *distributions* use that entropy
to sample a random number from a distribution. Distributions are function-like
objects whose constructors take the *parameters* of the distribution: for
example, a uniform distribution over the range :math:`[a, b)` takes the *a* and
*b* parameters as constructor arguments. The templated call operator accepts a
random engine as its sole argument.

Celeritas extends this paradigm to physics distributions. At a low level,
it has :ref:`random number distributions <celeritas_random>` that result in
single real values (such as uniform, exponential, gamma) and correlated
three-vectors (such as sampling an isotropic direction).

.. doxygenclass:: celeritas::BernoulliDistribution
   :members: none
.. doxygenclass:: celeritas::DeltaDistribution
   :members: none
.. doxygenclass:: celeritas::ExponentialDistribution
   :members: none
.. doxygenclass:: celeritas::GammaDistribution
   :members: none
.. doxygenclass:: celeritas::IsotropicDistribution
   :members: none
.. doxygenclass:: celeritas::NormalDistribution
   :members: none
.. doxygenclass:: celeritas::PoissonDistribution
   :members: none
.. doxygenclass:: celeritas::RadialDistribution
   :members: none
.. doxygenclass:: celeritas::ReciprocalDistribution
   :members: none
.. doxygenclass:: celeritas::UniformBoxDistribution
   :members: none
.. doxygenclass:: celeritas::UniformRealDistribution
   :members: none

.. _celeritas_physics:

Physics distributions
---------------------

At a higher level, Celeritas expresses many physics operations as
distributions of *updated* track states based on *original* track states. For
example, the Tsai-Urban distribution used for sampling exiting angles of
bremsstrahlung and pair production has parameters of incident particle energy
and mass, and it samples the exiting polar angle cosine.

.. doxygenclass:: celeritas::BhabhaEnergyDistribution
   :members: none

.. doxygenclass:: celeritas::EnergyLossGammaDistribution
   :members: none

.. doxygenclass:: celeritas::EnergyLossGaussianDistribution
   :members: none

.. doxygenclass:: celeritas::EnergyLossUrbanDistribution
   :members: none

.. doxygenclass:: celeritas::MollerEnergyDistribution
   :members: none

.. doxygenclass:: celeritas::TsaiUrbanDistribution
   :members: none


Physics implementations
-----------------------

Additional distributions are built on top of the helper distributions above.
All discrete interactions (in Geant4 parlance, "post-step do-it"s) use
distributions to sample an *Interaction* based on incident particle properties.
The sampled result contains the updated particle direction and energy, as well
as properties of any secondary particles produced.

.. doxygenclass:: celeritas::BetheHeitlerInteractor
   :members: none
.. doxygenclass:: celeritas::EPlusGGInteractor
   :members: none
.. doxygenclass:: celeritas::KleinNishinaInteractor
   :members: none
.. doxygenclass:: celeritas::MollerBhabhaInteractor
   :members: none
.. doxygenclass:: celeritas::LivermorePEInteractor
   :members: none
.. doxygenclass:: celeritas::RayleighInteractor
   :members: none
.. doxygenclass:: celeritas::RelativisticBremInteractor
   :members: none
.. doxygenclass:: celeritas::SeltzerBergerInteractor
   :members: none

.. doxygenclass:: celeritas::AtomicRelaxation
   :members: none
.. doxygenclass:: celeritas::EnergyLossHelper
   :members: none
.. doxygenclass:: celeritas::detail::UrbanMscStepLimit
   :members: none
.. doxygenclass:: celeritas::detail::UrbanMscScatter
   :members: none

.. _api_importdata:

Physics data
------------

Celeritas reads physics data from Geant4 (or from a ROOT file exported from
data previously loaded into Geant4). Different versions of Geant4 (and Geant4
data) can be used seamlessly with any version of Celeritas, allowing
differences to be isolated without respect to machine or model implementation.
The following classes enumerate all the data used at runtime.

.. doxygenstruct:: celeritas::ImportData
   :undoc-members:

Material and geometry properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::ImportElement
   :undoc-members:
.. doxygenstruct:: celeritas::ImportProductionCut
   :undoc-members:
.. doxygenstruct:: celeritas::ImportMatElemComponent
   :undoc-members:
.. doxygenstruct:: celeritas::ImportMaterial
   :undoc-members:
.. doxygenstruct:: celeritas::ImportVolume
   :undoc-members:
.. doxygenstruct:: celeritas::ImportTransParameters
   :undoc-members:
.. doxygenstruct:: celeritas::ImportLoopingThreshold
   :undoc-members:

.. doxygenenum:: ImportMaterialState

Physics properties
~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::ImportParticle
   :undoc-members:
.. doxygenstruct:: celeritas::ImportProcess
   :undoc-members:
.. doxygenstruct:: celeritas::ImportModel
   :undoc-members:
.. doxygenstruct:: celeritas::ImportMscModel
   :undoc-members:
.. doxygenstruct:: celeritas::ImportModelMaterial
   :undoc-members:
.. doxygenstruct:: celeritas::ImportPhysicsTable
   :undoc-members:
.. doxygenstruct:: celeritas::ImportPhysicsVector
   :undoc-members:

.. doxygenenum:: ImportProcessType
.. doxygenenum:: ImportProcessClass
.. doxygenenum:: ImportModelClass
.. doxygenenum:: ImportTableType
.. doxygenenum:: ImportUnits
.. doxygenenum:: ImportPhysicsVectorType

EM data
~~~~~~~

.. doxygenstruct:: celeritas::ImportEmParameters
   :undoc-members:
.. doxygenstruct:: celeritas::ImportAtomicTransition
   :undoc-members:
.. doxygenstruct:: celeritas::ImportAtomicSubshell
   :undoc-members:
.. doxygenstruct:: celeritas::ImportAtomicRelaxation
   :undoc-members:

.. doxygenstruct:: celeritas::ImportLivermoreSubshell
   :undoc-members:
.. doxygenstruct:: celeritas::ImportLivermorePE
   :undoc-members:

.. doxygenstruct:: celeritas::ImportSBTable
   :undoc-members:
