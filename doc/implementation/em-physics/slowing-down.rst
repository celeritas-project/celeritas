.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _em_slowing_down:

Continuous slowing down
=======================

Most charged interactions emit one or more low-energy particles during their
interaction. Instead of creating explicit daughter tracks that are
immediately killed due to low energy, part of the interaction cross section is
lumped into a "slowing down" term that continuously deposits energy locally
over the step.

.. doxygenfunction:: celeritas::calc_mean_energy_loss

Energy loss fluctuations
------------------------

Since true energy loss is a stochastic function of many small collisions, the
*mean* energy loss term is an approximation. Additional
models are implemented to adjust the loss per step with stochastic sampling for
improved accuracy.

.. doxygenclass:: celeritas::EnergyLossHelper
.. doxygenclass:: celeritas::EnergyLossGammaDistribution
.. doxygenclass:: celeritas::EnergyLossGaussianDistribution
.. doxygenclass:: celeritas::EnergyLossUrbanDistribution
