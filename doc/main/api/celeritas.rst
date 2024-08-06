.. Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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

.. doxygenfile:: celeritas/Units.hh
   :sections: user-defined var innernamespace

.. _api_constants:

.. doxygennamespace:: constants

.. doxygenfile:: celeritas/Constants.hh
   :sections: user-defined var innernamespace

.. doxygenfile:: celeritas/Quantities.hh
   :sections: user-defined var innernamespace

Problem definition
------------------

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams


.. _api_stepping:

Stepping mechanics
------------------

.. doxygenenum:: celeritas::TrackStatus

.. doxygenenum:: celeritas::ActionOrder

.. doxygenclass:: celeritas::Stepper

Geant4 physics interfaces
-------------------------

.. doxygenclass:: celeritas::GeantImporter

.. doxygenclass:: celeritas::GeantSetup

.. _api_geant4_physics_options:

Geant4 physics options
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::GeantPhysicsOptions
   :members:

On-device access
----------------

.. doxygenclass:: celeritas::MaterialTrackView

.. doxygenclass:: celeritas::ParticleTrackView

.. doxygenclass:: celeritas::PhysicsTrackView

Propagation and magnetic field
------------------------------

The propagation interface is built on top of the geometry to allow both curved
and straight-line movement. Field propagation is based on a composition of:

Field
  Maps a point in space and time to a field vector.
Equation of motion
  Calculates the path derivative of position and momentum given their current
  state and the templated field.
Integrator
  Numerically integrates a new position/momentum state given the start,
  path derivative, and step length.
Driver
  Integrate path segments that satisfy certain error conditions, solving for
  the required segment length.
Propagator
  Given a maximum physics step, advance the geometry state and momentum along
  the field lines, satisfying constraints (see :ref:`field driver
  options<api_field_data>`) for the maximum geometry error.

Propagation
~~~~~~~~~~~

.. doxygenclass:: celeritas::LinearPropagator

.. doxygenclass:: celeritas::FieldPropagator

.. doxygenfunction:: celeritas::make_mag_field_propagator

.. _api_field_data:

Field data input and options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::FieldDriverOptions
   :members:

Field data
~~~~~~~~~~

These classes correspond to JSON input files to the field setup.

.. doxygenstruct:: celeritas::UniformFieldParams
   :members:

.. doxygenstruct:: celeritas::RZMapFieldInput
   :members:

.. _celeritas_random:

Random number generation
------------------------

The 2011 ISO C++ standard defined a new functional paradigm for sampling from
random number distributions. In this paradigm, random number *engines* generate
a uniformly distributed stream of bits. Then, *distributions* use that entropy
to sample a random number from a distribution.

Engines
~~~~~~~

Celeritas defaults to using an in-house implementation of the XORWOW
:cite:`marsaglia_xorshift_2003` bit shifting generator. Each thread's state is
seeded at runtime by filling the state with bits generated from a 32-bit
Mersenne twister. When a new event begins through the Geant4 interface, each
thread's state is initialized using same seed and skipped ahead a different
number of subsequences so the sequences on different threads will not have
statistically correlated values.

.. doxygenfunction:: celeritas::initialize_xorwow

.. doxygenclass:: celeritas::XorwowRngEngine

Distributions
~~~~~~~~~~~~~

Distributions are function-like
objects whose constructors take the *parameters* of the distribution: for
example, a uniform distribution over the range :math:`[a, b)` takes the *a* and
*b* parameters as constructor arguments. The templated call operator accepts a
random engine as its sole argument.

Celeritas extends this paradigm to physics distributions. At a low level,
it has :ref:`random number distributions <celeritas_random>` that result in
single real values (such as uniform, exponential, gamma) and correlated
three-vectors (such as sampling an isotropic direction).

.. doxygenclass:: celeritas::BernoulliDistribution
.. doxygenclass:: celeritas::DeltaDistribution
.. doxygenclass:: celeritas::ExponentialDistribution
.. doxygenclass:: celeritas::GammaDistribution
.. doxygenclass:: celeritas::IsotropicDistribution
.. doxygenclass:: celeritas::NormalDistribution
.. doxygenclass:: celeritas::PoissonDistribution
.. doxygenclass:: celeritas::RadialDistribution
.. doxygenclass:: celeritas::ReciprocalDistribution
.. doxygenclass:: celeritas::UniformBoxDistribution
.. doxygenclass:: celeritas::UniformRealDistribution

Additionally we define a few helper classes for common physics sampling
routines.

.. doxygenclass:: celeritas::RejectionSampler
.. doxygenclass:: celeritas::ElementSelector
.. doxygenclass:: celeritas::IsotopeSelector
.. doxygenclass:: celeritas::TabulatedElementSelector

.. _api_em_physics:

EM physics
----------

The physics models in Celeritas are primarily derived from references cited by Geant4,
including the Geant4 physics reference manual. Undocumented adjustments to
those models in Geant4 may also be implemented, and hopefully, explained in our
documentation.

Distributions
~~~~~~~~~~~~~

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


Implementations
~~~~~~~~~~~~~~~

Additional distributions are built on top of the helper distributions above.
All discrete interactions (in Geant4 parlance, "post-step do-it"s) use
distributions to sample an *Interaction* based on incident particle properties.
The sampled result contains the updated particle direction and energy, as well
as properties of any secondary particles produced.

.. doxygenclass:: celeritas::BetheHeitlerInteractor
.. doxygenclass:: celeritas::BraggICRU73QOInteractor
.. doxygenclass:: celeritas::CoulombScatteringInteractor
.. doxygenclass:: celeritas::EPlusGGInteractor
.. doxygenclass:: celeritas::KleinNishinaInteractor
.. doxygenclass:: celeritas::MollerBhabhaInteractor
.. doxygenclass:: celeritas::LivermorePEInteractor
.. doxygenclass:: celeritas::MuBetheBlochInteractor
.. doxygenclass:: celeritas::MuBremsstrahlungInteractor
.. doxygenclass:: celeritas::RayleighInteractor
.. doxygenclass:: celeritas::RelativisticBremInteractor
.. doxygenclass:: celeritas::SeltzerBergerInteractor

.. doxygenclass:: celeritas::AtomicRelaxation
.. doxygenclass:: celeritas::EnergyLossHelper
.. doxygenclass:: celeritas::detail::UrbanMscSafetyStepLimit
.. doxygenclass:: celeritas::detail::UrbanMscScatter

.. doxygenclass:: celeritas::SBEnergyDistribution
.. doxygenclass:: celeritas::detail::SBPositronXsCorrector

.. _api_importdata:

Physics data
------------

Celeritas reads physics data from Geant4 (or from a ROOT file exported from
data previously loaded into Geant4). Different versions of Geant4 (and Geant4
data) can be used seamlessly with any version of Celeritas, allowing
differences to be isolated without respect to machine or model implementation.
The following classes enumerate all the data used at runtime.

.. doxygenstruct:: celeritas::ImportData
   :members:
   :undoc-members:

Material and geometry properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::ImportIsotope
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportElement
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportMatElemComponent
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportGeoMaterial
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportProductionCut
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportPhysMaterial
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportRegion
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportVolume
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportTransParameters
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportLoopingThreshold
   :members:
   :undoc-members:

.. doxygenenum:: ImportMaterialState

Physics properties
~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: celeritas::ImportParticle
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportProcess
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportModel
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportMscModel
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportModelMaterial
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportPhysicsTable
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportPhysicsVector
   :members:
   :undoc-members:

.. doxygenenum:: ImportUnits

EM data
~~~~~~~

.. doxygenstruct:: celeritas::ImportEmParameters
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportAtomicTransition
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportAtomicSubshell
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportAtomicRelaxation
   :members:
   :undoc-members:

.. doxygenstruct:: celeritas::ImportLivermoreSubshell
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportLivermorePE
   :members:
   :undoc-members:

.. doxygenstruct:: celeritas::ImportSBTable
   :members:
   :undoc-members:

Optical data
~~~~~~~~~~~~

.. doxygenstruct:: celeritas::ImportOpticalAbsorption
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportOpticalMaterial
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportOpticalParameters
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportOpticalProperty
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportOpticalRayleigh
   :members:
   :undoc-members:

.. doxygenstruct:: celeritas::ImportScintComponent
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportScintData
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportParticleScintSpectrum
   :members:
   :undoc-members:
.. doxygenstruct:: celeritas::ImportMaterialScintSpectrum
   :members:
   :undoc-members:

.. doxygenstruct:: celeritas::ImportWavelengthShift
   :members:
   :undoc-members:

