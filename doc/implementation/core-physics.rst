.. Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_celeritas:

************
Core physics
************

The core of ``celeritas`` provides the physics and stepping loop
implementation for the Celeritas codebase.


Problem definition
==================

Celeritas contains several high-level "parameter" classes that allow setup-time
access to problem data. These classes all correspond directly to "TrackView"
classes (see the `developer documentation`_ for details).

.. _developer documentation: https://celeritas-project.github.io/celeritas/dev/classes.html

.. doxygenclass:: celeritas::GeoParamsInterface

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams

.. doxygenclass:: celeritas::CutoffParams


.. _api_stepping:

Stepping mechanics
==================

.. doxygenenum:: celeritas::TrackStatus
   :no-link:

.. doxygenenum:: celeritas::ActionOrder
   :no-link:

.. doxygenclass:: celeritas::Stepper


Propagation and magnetic field
==============================

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
-----------

.. doxygenclass:: celeritas::LinearPropagator

.. doxygenclass:: celeritas::FieldPropagator

.. doxygenfunction:: celeritas::make_mag_field_propagator


.. _api_field_data:

Field data input and options
----------------------------

These classes correspond to JSON input files to the field setup.

.. doxygenstruct:: celeritas::UniformFieldParams
   :members:
   :no-link:

.. doxygenstruct:: celeritas::RZMapFieldInput
   :members:
   :no-link:


The field driver options are not yet a stable part of the API:

.. doxygenstruct:: celeritas::FieldDriverOptions
   :members:
   :no-link:

.. _celeritas_random:

Random number generation
========================

The 2011 ISO C++ standard defined a new functional paradigm for sampling from
random number distributions. In this paradigm, random number *engines* generate
a uniformly distributed stream of bits. Then, *distributions* use that entropy
to sample a random number from a distribution.

Engines
-------

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
-------------

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
.. doxygenclass:: celeritas::InverseSquareDistribution
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


.. _api_importdata:

Imported data
=============

Celeritas reads physics data from Geant4 (or from a ROOT file exported from
data previously loaded into Geant4). Different versions of Geant4 (and Geant4
data) can be used seamlessly with any version of Celeritas, allowing
differences to be isolated without respect to machine or model implementation.
The following classes enumerate the core data loaded at runtime.

.. doxygenstruct:: celeritas::ImportData
   :members:
   :undoc-members:
   :no-link:

Material and geometry properties
--------------------------------

.. doxygenstruct:: celeritas::ImportIsotope
.. doxygenstruct:: celeritas::ImportElement
.. doxygenstruct:: celeritas::ImportMatElemComponent
.. doxygenstruct:: celeritas::ImportGeoMaterial
.. doxygenstruct:: celeritas::ImportProductionCut
.. doxygenstruct:: celeritas::ImportPhysMaterial
.. doxygenstruct:: celeritas::ImportRegion
.. doxygenstruct:: celeritas::ImportVolume
.. doxygenstruct:: celeritas::ImportTransParameters
.. doxygenstruct:: celeritas::ImportLoopingThreshold

.. doxygenenum:: ImportMaterialState
   :no-link:

Physics properties
------------------

.. doxygenstruct:: celeritas::ImportParticle
.. doxygenstruct:: celeritas::ImportProcess
.. doxygenstruct:: celeritas::ImportModel
.. doxygenstruct:: celeritas::ImportMscModel
.. doxygenstruct:: celeritas::ImportModelMaterial
.. doxygenstruct:: celeritas::ImportPhysicsTable
.. doxygenstruct:: celeritas::ImportPhysicsVector

.. doxygenenum:: ImportUnits
   :no-link:
