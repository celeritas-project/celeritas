.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _em_ionization:

Ionization
==========

Models
------

.. doxygenclass:: celeritas::MollerBhabhaInteractor
.. doxygenclass:: celeritas::MuHadIonizationInteractor

Distributions
-------------

The exiting energy distribution from most of these ionization models
are sampled using external helper distributions.

.. doxygenclass:: celeritas::BetheBlochEnergyDistribution
.. doxygenclass:: celeritas::BraggICRU73QOEnergyDistribution
.. doxygenclass:: celeritas::BhabhaEnergyDistribution
.. doxygenclass:: celeritas::MollerEnergyDistribution
.. doxygenclass:: celeritas::MuBBEnergyDistribution

