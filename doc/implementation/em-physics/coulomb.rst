.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _em_coulomb:

Coulomb scattering
==================

Elastic scattering of charged particles off atoms can be simulated in three ways:

* A detailed single scattering model in which each scattering interaction is
  sampled
* A multiple scattering approach which calculates global effects from many
  collisions
* A combination of the two

Though it is the most accurate, the single Coulomb scattering model is too
computationally expensive to be used in most applications as the number of
collisions can be extremely large. Instead, a "condensed" simulation algorithm
is typically used to determine the net energy loss, displacement, and direction
change from many collisions after a given path length. The Urban model is the
default multiple scattering model in Celeritas for all energies and in Geant4
below 100 MeV. A third "mixed" simulation approach uses multiple scattering to
simulate interactions with scattering angles below a given polar angle limit
and single scattering for large angles. The Wentzel-VI model, used together
with the single Coulomb scattering model, is an implementation of the mixed
simulation algorithm. It is the default model in Geant4 above 100 MeV and
currently under development in Celeritas.

Models
------

.. doxygenclass:: celeritas::CoulombScatteringInteractor
.. doxygenclass:: celeritas::detail::UrbanMscScatter

Cross sections
--------------

.. doxygenclass:: celeritas::WentzelHelper
.. doxygenclass:: celeritas::MottRatioCalculator

Distributions
-------------

.. doxygenclass:: celeritas::WentzelDistribution

The nuclear form factors used by this distribution are:

.. doxygenclass:: celeritas::ExpNuclearFormFactor
.. doxygenclass:: celeritas::GaussianNuclearFormFactor
.. doxygenclass:: celeritas::UUNuclearFormFactor
