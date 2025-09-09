.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _em_brems:

Bremsstrahlung
==============

Models
------

.. doxygenclass:: celeritas::RelativisticBremInteractor
.. doxygenclass:: celeritas::SeltzerBergerInteractor
.. doxygenclass:: celeritas::MuBremsstrahlungInteractor

The exiting electron energies from the interaction are calculated with a shared
helper function:

.. doxygenclass:: celeritas::detail::BremFinalStateHelper

Cross sections
--------------

The bremsstrahlung interactions are sampled using cross sections and adjustment factors:

.. doxygenclass:: celeritas::SBEnergyDistribution
.. doxygenclass:: celeritas::detail::SBPositronXsCorrector

.. doxygenclass:: celeritas::RBDiffXsCalculator

Relativistic bremsstrahlung and relativistic Bethe-Heitler sampling both use a
helper class to calculate a suppression factor that reduces the differential
cross section for low-energy gammas.

.. doxygenclass:: celeritas::LPMCalculator

They also use nuclear screening functions:

.. doxygenclass:: celeritas::TsaiScreeningCalculator

Muon bremsstrahlung calculates the differential cross section as part of
rejection sampling.

.. doxygenclass:: celeritas::MuBremsDiffXsCalculator

Distributions
-------------

.. doxygenclass:: celeritas::detail::SBEnergySampler
.. doxygenclass:: celeritas::detail::RBEnergySampler

A simple distribution is used to sample exiting polar angles from electron
bremsstrahlung (and gamma conversion).

.. doxygenclass:: celeritas::TsaiUrbanDistribution

Muon bremsstrahlung and pair production use a simple distribution to sample the
exiting polar angles.

.. doxygenclass:: celeritas::MuAngularDistribution
