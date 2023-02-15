.. Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _overview:

********
Overview
********

This section summarizes key usage patterns and implementation details from
Celeritas, especially those that differ from Geant4.

Units
=====

The unit system in Celeritas is CGS with centimeter (cm), gram (g), second (s),
gauss (Ga?), and kelvin (K) all having a value of unity. With these definitions,
densities can be defined in natural units of :math:`\mathrm{g}/\mathrm{cm}^3`,
and macroscopic cross sections are in units of :math:`\mathrm{cm}^{-1}`. See
the :ref:`units documentation <api_units>` for more descriptions of the core
unit system and the exactly defined values for SI units such as tesla.

Celeritas defines :ref:`constants <api_constants>` from a few different sources.
Mathematical constants are defined as truncated floating point values. Some
physical constants such as the speed of light, Planck's constant, and the
electron charge, have exact numerical values as specified by the SI unit system
:cite:`SI`. Other physical constants such as the atomic mass unit and electron
radius are derived from experimental measurements in CODATA 2018. Because the
reported constants are derived from regression fits to experimental data
points, some exactly defined physical relationships (such as the fine structure
:math:`\alpha = \frac{e^2}{2 \epsilon_0 h c}` are only approximate.

Unlike Geant4 and the CLHEP unit systems, Celeritas avoids using "natural"
units in its definitions. Although a natural unit system makes some
expressions easier to evaluate, it can lead to errors in the definition of
derivative constants and is fundamentally in conflict with consistent unit
systems such as SI. To enable special unit systems in harmony with the
native Celeritas unit system, the :ref:`Quantity <api_quantity>` class
stores quantities in another unit system with a compile-time constant that
allows their conversion back to native units. This allows, for example,
particles to represent their energy as MeV and charge as fractions of e but
work seamlessly with a field definition in native (macro-scale quantity) units.

Distributions
=============

The 2011 ISO C++ standard defined a new functional paradigm for sampling from
random number distributions. In this paradigm, random number *engines* generate
a uniformly distributed stream of bits. Then, *distributions* use that entropy
to sample a random number from a distribution. Distributions are function-like
objects whose constructors take the *parameters* of the distribution: for
example, a uniform distribution over the range :math:`[a, b)` takes the *a* and
*b* parameters as constructor arguments. The templated call operator accepts a
random engine as its sole argument.

Celeritas extends this paradigm to physics distributions. At a low level,
it has :ref:`random number distributions <celeritas_random>` that result in
single real values (such as uniform, exponential, gamma) and correlated
three-vectors (such as sampling an isotropic direction).

At a higher level, though, it expresses many physics operations as
distributions of *updated* track states based on *original* track states. For
example, the Tsai-Urban distribution used for sampling exiting angles of
bremsstrahlung and pair production has parameters of incident particle energy
and mass, and it samples the exiting polar angle cosine. Additional
distributions are built on top of those: all discrete interactions (in Geant4
parlance, "post-step do-it"s) use distributions to sample an *Interaction*
based on incident particle properties. The sampled result contains the updated
particle direction and energy, as well as properties of any secondary particles
produced.

Physics
=======

.. table:: Electromagnetic physics processes and models available in Celeritas and their equivalents in Geant4.

   +----------------+---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   | **Particle**   | **Processes**       | **Models**                | **Geant4 (EM-Opt0 Physics List)**  | **Celeritas**                                 | **Applicability**        |
   +----------------+---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |:math:`e^-`     | Ionisation          | Moller                    | ``G4MollerBhabhaModel``            | :cpp:class:`celeritas::MollerBhabhaModel`     |       0 - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Bremsstrahlung      | Seltzer-Berger            | ``G4SeltzerBergerModel``           | :cpp:class:`celeritas::SeltzerBergerModel`    |       0 -   1 GeV        |
   |                |                     +---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                |                     | Relativistic              | ``G4eBremsstrahlungRelModel``      | :cpp:class:`celeritas::RelativisticBremModel` |   1 GeV - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Coulomb scattering  | Urban                     | ``G4UrbanMscModel``                | :cpp:class:`celeritas::UrbanMsc` [1]_         |   10 eV - 100 MeV        |
   +----------------+---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |:math:`e^+`     | Ionisation          | Bhabha                    | ``G4MollerBhabhaModel``            | :cpp:class:`celeritas::MollerBhabhaModel`     |       0 - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Bremsstrahlung      | Seltzer-Berger            | ``G4SeltzerBergerModel``           | :cpp:class:`celeritas::SeltzerBergerModel`    |       0 -   1 GeV        |
   |                |                     +---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                |                     | Relativistic              | ``G4eBremsstrahlungRelModel``      | :cpp:class:`celeritas::RelativisticBremModel` |   1 GeV - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Coulomb scattering  | Urban                     | ``G4UrbanMscModel``                | :cpp:class:`celeritas::UrbanMsc` [1]_         |   10 eV - 100 MeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Annihilation        |:math:`e^+-e^- \to 2\gamma`| ``G4eplusAnnihilation``            | :cpp:class:`celeritas::EPlusGGModel`          |       0 - 100 TeV        |
   +----------------+---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |:math:`\gamma`  | Photoelectric       | Livermore                 | ``G4LivermorePhotoElectricModel``  | :cpp:class:`celeritas::LivermorePEModel`      |       0 - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Compton scattering  | Klein - Nishina           | ``G4KleinNishinaCompton``          | :cpp:class:`celeritas::KleinNishinaModel`     |       0 - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Pair production     | Bethe - Heitler           | ``G4PairProductionRelModel``       | :cpp:class:`celeritas::BetheHeitlerModel`     |       0 - 100 TeV        |
   |                +---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+
   |                | Rayleigh scattering | Livermore                 | ``G4LivermoreRayleighModel``       | :cpp:class:`celeritas::RayleighModel`         |       0 - 100 TeV        |
   +----------------+---------------------+---------------------------+------------------------------------+-----------------------------------------------+--------------------------+

.. [1] Multiple Scattering using the Urban Model is only applied up to 100MeV in Celeritas, with no model used above this energy.  

Geometry
========

Celeritas has two choices of geometry implementation. VecGeom_ is a
CUDA-compatible library for navigation on Geant4 detector geometries.
:ref:`api_orange` is a work in progress for surface-based geometry navigation
that is "platform portable", i.e. able to run on GPUs from multiple vendors.

Celeritas wraps both geometry packages with a uniform interface for changing
and querying the geometry state.

.. _VecGeom: https://gitlab.cern.ch/VecGeom/VecGeom

Stepping loop
=============

The stepping loop in Celeritas is a sorted loop over "actions", each of which
is usually a kernel launch (or an inner loop over tracks if running on CPU).

GPU usage
=========

Celeritas automatically copies data to device when constructing objects as long
as the GPU is enabled.
