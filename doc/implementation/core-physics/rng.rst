.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

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
:cite:`marsaglia-xorshift-2003` bit shifting generator. Each thread's state is
seeded at runtime by filling the state with bits generated from a 32-bit
Mersenne twister. When a new event begins through the Geant4 interface, each
thread's state is initialized using same seed and skipped ahead a different
number of subsequences so the sequences on different threads will not have
statistically correlated values.

.. doxygenfunction:: celeritas::initialize_xorwow

.. doxygenclass:: celeritas::XorwowRngEngine

.. _celeritas_random_distributions:

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
.. doxygenclass:: celeritas::PowerDistribution
.. doxygenclass:: celeritas::RadialDistribution
.. doxygenclass:: celeritas::ReciprocalDistribution
.. doxygenclass:: celeritas::UniformBoxDistribution
.. doxygenclass:: celeritas::UniformRealDistribution

Additionally we define a few helper classes for common physics sampling
routines.

.. doxygenclass:: celeritas::RejectionSampler
.. doxygenclass:: celeritas::Selector
.. doxygenfunction:: celeritas::make_selector
.. doxygenfunction:: celeritas::make_unnormalized_selector

And specifically for elements and isotopes:

.. doxygenfunction:: celeritas::make_isotope_selector
.. doxygenclass:: celeritas::ElementSelector
.. doxygenclass:: celeritas::TabulatedElementSelector

The physics model implementations are built on top of these helper
distributions.
