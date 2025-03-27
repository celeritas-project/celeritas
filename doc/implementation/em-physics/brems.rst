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

Cross sections
--------------

The Seltzer--Berger interactions are sampled with the help of an energy
distribution and cross section correction:

.. doxygenclass:: celeritas::SBEnergyDistribution
.. doxygenclass:: celeritas::detail::SBPositronXsCorrector

Relativistic bremsstrahlung and relativistic Bethe-Heitler sampling both use a
helper class to calculate a suppression factor that reduces the differential
cross section for low-energy gammas.

.. doxygenclass:: celeritas::LPMCalculator

Muon bremsstrahlung calculates the differential cross section as part of
rejection sampling.

.. doxygenclass:: celeritas::MuBremsDiffXsCalculator

Distributions
-------------

A simple distribution is used to sample exiting polar angles from electron
bremsstrahlung (and gamma conversion).

.. doxygenclass:: celeritas::TsaiUrbanDistribution

Muon bremsstrahlung and pair production use a simple distribution to sample the
exiting polar angles.

.. doxygenclass:: celeritas::MuAngularDistribution


