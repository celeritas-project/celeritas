.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _em_pair-production:

Pair production
===============

Models
------

.. doxygenclass:: celeritas::BetheHeitlerInteractor
.. doxygenclass:: celeritas::MuPairProductionInteractor

Distributions
-------------

The energy transfer for muon pair production is sampled using the inverse
transform method with tabulated CDFs.

.. doxygenclass:: celeritas::MuPPEnergyDistribution

