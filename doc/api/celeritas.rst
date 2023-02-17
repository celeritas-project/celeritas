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

External code interfaces
------------------------

.. doxygenclass:: celeritas::GeantImporter

.. doxygenclass:: celeritas::GeantSetup

.. doxygenclass:: celeritas::VecgeomParams

On-device access
----------------

.. doxygenclass:: celeritas::MaterialTrackView

.. doxygenclass:: celeritas::ParticleTrackView

.. doxygenclass:: celeritas::PhysicsTrackView


.. _celeritas_random:

Random number distributions
---------------------------

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

Physics interfaces
------------------

.. doxygenclass:: celeritas::BetheHeitlerModel
   :members: none
.. doxygenclass:: celeritas::EPlusGGModel
   :members: none
.. doxygenclass:: celeritas::KleinNishinaModel
   :members: none
.. doxygenclass:: celeritas::MollerBhabhaModel
   :members: none
.. doxygenclass:: celeritas::LivermorePEModel
   :members: none
.. doxygenclass:: celeritas::RayleighModel
   :members: none
.. doxygenclass:: celeritas::RelativisticBremModel
   :members: none
.. doxygenclass:: celeritas::SeltzerBergerModel
   :members: none

.. doxygenclass:: celeritas::AtomicRelaxationParams
   :members: none
.. doxygenclass:: celeritas::FluctuationParams
   :members: none
.. doxygenclass:: celeritas::UrbanMscParams
   :members: none
