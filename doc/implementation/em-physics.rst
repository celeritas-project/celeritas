.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_em_physics:

**********
EM Physics
**********

The physics models in Celeritas are primarily derived from references cited by
Geant4, including the Geant4 physics reference manual. Undocumented adjustments
to those models in Geant4 may also be implemented.

Processes and models
====================

The following table summarizes the EM processes and models in Celeritas.

.. only:: html

   .. table:: Electromagnetic physics processes and models available in Celeritas.

      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | Particle       | Processes           |  Models                     | Celeritas Implementation                            | Applicability            |
      +================+=====================+=============================+=====================================================+==========================+
      | :math:`e^-`    | Ionization          |  Møller                     | :cpp:class:`celeritas::MollerBhabhaInteractor`      |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Bremsstrahlung      |  Seltzer--Berger            | :cpp:class:`celeritas::SeltzerBergerInteractor`     |       0--1 GeV           |
      |                |                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Relativistic               | :cpp:class:`celeritas::RelativisticBremInteractor`  |   1 GeV -- 100 TeV       |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Coulomb scattering  |  Urban                      | :cpp:class:`celeritas::UrbanMscScatter`             |   100 eV -- 100 TeV      |
      |                |                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Coulomb                    | :cpp:class:`celeritas::CoulombScatteringInteractor` |       0--100 TeV         |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | :math:`e^+`    | Ionization          |  Bhabha                     | :cpp:class:`celeritas::MollerBhabhaInteractor`      |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Bremsstrahlung      |  Seltzer-Berger             | :cpp:class:`celeritas::SeltzerBergerInteractor`     |       0--1 GeV           |
      |                |                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Relativistic               | :cpp:class:`celeritas::RelativisticBremInteractor`  |   1 GeV -- 100 TeV       |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Coulomb scattering  |  Urban                      | :cpp:class:`celeritas::UrbanMscScatter`             |   100 eV -- 100 TeV      |
      |                |                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Coulomb                    | :cpp:class:`celeritas::CoulombScatteringInteractor` |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Annihilation        | :math:`e^+,e^- \to 2\gamma` | :cpp:class:`celeritas::EPlusGGInteractor`           |       0--100 TeV         |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | :math:`\gamma` | Photoelectric       |  Livermore                  | :cpp:class:`celeritas::LivermorePEInteractor`       |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Compton scattering  |  Klein--Nishina             | :cpp:class:`celeritas::KleinNishinaInteractor`      |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Pair production     |  Bethe--Heitler             | :cpp:class:`celeritas::BetheHeitlerInteractor`      |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Rayleigh scattering |  Livermore                  | :cpp:class:`celeritas::RayleighInteractor`          |       0--100 TeV         |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | :math:`\mu^-`  | Ionization          |  ICRU73QO                   | :cpp:class:`celeritas::MuHadIonizationInteractor`   |       0--200 keV         |
      |                +                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Bethe--Bloch               | :cpp:class:`celeritas::MuHadIonizationInteractor`   |   200 keV--1 GeV         |
      |                +                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Mu Bethe--Bloch            | :cpp:class:`celeritas::MuHadIonizationInteractor`   |   200 keV--100 TeV       |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Bremsstrahlung      |  Mu bremsstrahlung          | :cpp:class:`celeritas::MuBremsstrahlungInteractor`  |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Pair production     |  Mu pair production         | :cpp:class:`celeritas::MuPairProductionInteractor`  |   0.85 GeV--100 TeV      |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | :math:`\mu^+`  | Ionization          |  Bragg                      | :cpp:class:`celeritas::MuHadIonizationInteractor`   |       0--200 keV         |
      |                +                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Bethe--Bloch               | :cpp:class:`celeritas::MuHadIonizationInteractor`   |   200 keV--1 GeV         |
      |                +                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Mu Bethe--Bloch            | :cpp:class:`celeritas::MuHadIonizationInteractor`   |   200 keV--100 TeV       |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Bremsstrahlung      |  Mu bremsstrahlung          | :cpp:class:`celeritas::MuBremsstrahlungInteractor`  |       0--100 TeV         |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Pair production     |  Mu pair production         | :cpp:class:`celeritas::MuPairProductionInteractor`  |   0.85 GeV--100 TeV      |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+

.. only:: latex

   .. raw:: latex

      \begin{table}[h]
        \caption{Electromagnetic physics processes and models available in Celeritas.}
        \begin{threeparttable}
        \begin{tabular}{llllr}
          \toprule
          Particle         & Processes                  & Models      & Celeritas Implementation                           & Applicability \\
          \midrule
          \multirow{4}{*}{$e^-$}    & Ionization                          & Møller               & \texttt{\scriptsize celeritas::MollerBhabhaInteractor}      & 0--100 TeV \\
                                    \cline{2-5}
                                    & \multirow{2}{*}{Bremsstrahlung}     & Seltzer--Berger      & \texttt{\scriptsize celeritas::SeltzerBergerInteractor}     & 0--1 GeV \\
                                                                          \cline{3-5}
                                    &                                     & Relativistic         & \texttt{\scriptsize celeritas::RelativisticBremInteractor}  & 1 GeV -- 100 TeV \\
                                    \cline{2-5}
                                    & \multirow{2}{*}{Coulomb scattering} & Urban                & \texttt{\scriptsize celeritas::UrbanMscScatter}             & 100 eV -- 100 TeV \\
                                                                          \cline{3-5}
                                    &                                     & Coulomb              & \texttt{\scriptsize celeritas::CoulombScatteringInteractor} & 0--100 TeV \\
                                    \cline{2-5}
          \hline
          \multirow{5}{*}{$e^+$}    & Ionization                          & Bhabha               & \texttt{\scriptsize celeritas::MollerBhabhaInteractor}      & 0--100 TeV \\
                                    \cline{2-5}
                                    & \multirow{2}{*}{Bremsstrahlung}     & Seltzer--Berger      & \texttt{\scriptsize celeritas::SeltzerBergerInteractor}     & 0--1 GeV \\
                                                                          \cline{3-5}
                                    &                                     & Relativistic         & \texttt{\scriptsize celeritas::RelativisticBremInteractor}  & 1 GeV -- 100 TeV \\
                                    \cline{2-5}
                                    & \multirow{2}{*}{Coulomb scattering} & Urban                & \texttt{\scriptsize celeritas::UrbanMscScatter}             & 100 eV -- 100 TeV \\
                                                                          \cline{3-5}
                                    &                                     & Coulomb              & \texttt{\scriptsize celeritas::CoulombScatteringInteractor} & 0--100 TeV \\
                                    \cline{2-5}
                                    & Annihilation                        & $e^+,e^-\to 2\gamma$ & \texttt{\scriptsize celeritas::EPlusGGInteractor}           & 0--100 TeV \\
          \hline
          \multirow{4}{*}{$\gamma$} & Photoelectric                       & Livermore            & \texttt{\scriptsize celeritas::LivermorePEInteractor}       & 0--100 TeV \\
                                    \cline{2-5}
                                    & Compton scattering                  & Klein--Nishina       & \texttt{\scriptsize celeritas::KleinNishinaInteractor}      & 0--100 TeV \\
                                    \cline{2-5}
                                    & Pair production                     & Bethe--Heitler       & \texttt{\scriptsize celeritas::BetheHeitlerInteractor}      & 0--100 TeV \\
                                    \cline{2-5}
                                    & Rayleigh scattering                 & Livermore            & \texttt{\scriptsize celeritas::RayleighInteractor}          & 0--100 TeV \\
          \hline
          \multirow{3}{*}{$\mu^-$}  & \multirow{2}{*}{Ionization}         & ICRU73QO             & \texttt{\scriptsize celeritas::MuHadIonizationInteractor}   & 0--200 keV \\
                                                                          \cline{3-5}
                                    &                                     & Bethe--Bloch         & \texttt{\scriptsize celeritas::MuHadIonizationInteractor}   & 200 keV -- 1 GeV \\
                                                                          \cline{3-5}
                                    &                                     & Mu Bethe--Bloch      & \texttt{\scriptsize celeritas::MuHadIonizationInteractor}   & 200 keV -- 100 TeV \\
                                    \cline{2-5}
                                    & Bremsstrahlung                      & Mu bremsstrahlung    & \texttt{\scriptsize celeritas::MuBremsstrahlungInteractor}  & 0--100 TeV \\
                                    \cline{2-5}
                                    & Pair production                     & Mu pair production   & \texttt{\scriptsize celeritas::MuPairProductionInteractor}  & 0.85 GeV--100 TeV \\
          \hline
          \multirow{3}{*}{$\mu^+$}  & \multirow{2}{*}{Ionization}         & Bragg                & \texttt{\scriptsize celeritas::MuHadIonizationInteractor}   & 0--200 keV \\
                                                                          \cline{3-5}
                                    &                                     & Bethe--Bloch         & \texttt{\scriptsize celeritas::MuHadIonizationInteractor}   & 200 keV -- 1 GeV \\
                                                                          \cline{3-5}
                                    &                                     & Mu Bethe--Bloch      & \texttt{\scriptsize celeritas::MuHadIonizationInteractor}   & 200 keV -- 100 TeV \\
                                    \cline{2-5}
                                    & Bremsstrahlung                      & Mu bremsstrahlung    & \texttt{\scriptsize celeritas::MuBremsstrahlungInteractor}  & 0--100 TeV \\
                                    \cline{2-5}
                                    & Pair production                     & Mu pair production   & \texttt{\scriptsize celeritas::MuPairProductionInteractor}  & 0.85 GeV--100 TeV \\
          \bottomrule
        \end{tabular}
        \end{threeparttable}
      \end{table}

The implemented physics models are meant to match the defaults constructed in
``G4EmStandardPhysics``.  Known differences are:

* Particles other than electrons, positrons, and gammas are not currently
  supported.
* As with the AdePT project, Celeritas currently extends the range of Urban MSC
  to higher energies rather than implementing the Wentzel-VI and discrete
  Coulomb scattering.
* Celeritas imports tracking cutoffs and other parameters from
  ``G4EmParameters``, but some custom model cutoffs are not accessible to
  Celeritas.

As extension to the various :ref:`random distributions
<celeritas_random_distributions>`, Celeritas expresses many physics operations
as
distributions of *updated* track states based on *original* track states. For
example, the Tsai-Urban distribution used for sampling exiting angles of
bremsstrahlung and pair production has parameters of incident particle energy
and mass, and it samples the exiting polar angle cosine.

All discrete interactions (in Geant4 parlance, "post-step do-it"s) use
distributions to sample an *Interaction* based on incident particle
properties.
The sampled result contains the updated particle direction and energy, as well
as properties of any secondary particles produced.

Ionization
----------

.. doxygenclass:: celeritas::MollerBhabhaInteractor
.. doxygenclass:: celeritas::MuHadIonizationInteractor

The exiting energy distribution from most of these ionization models
are sampled using external helper distributions.

.. doxygenclass:: celeritas::BetheBlochEnergyDistribution
.. doxygenclass:: celeritas::BraggICRU73QOEnergyDistribution
.. doxygenclass:: celeritas::BhabhaEnergyDistribution
.. doxygenclass:: celeritas::MollerEnergyDistribution
.. doxygenclass:: celeritas::MuBBEnergyDistribution


Bremsstrahlung
--------------

.. doxygenclass:: celeritas::RelativisticBremInteractor
.. doxygenclass:: celeritas::SeltzerBergerInteractor
.. doxygenclass:: celeritas::MuBremsstrahlungInteractor


The Seltzer--Berger interactions are sampled with the help of an energy
distribution and cross section correction:

.. doxygenclass:: celeritas::SBEnergyDistribution
.. doxygenclass:: celeritas::detail::SBPositronXsCorrector

A simple distribution is used to sample exiting polar angles from electron
bremsstrahlung (and gamma conversion).

.. doxygenclass:: celeritas::TsaiUrbanDistribution

Relativistic bremsstrahlung and relativistic Bethe-Heitler sampling both use a
helper class to calculate LPM factors.

.. doxygenclass:: celeritas::LPMCalculator

Muon bremsstrahlung calculates the differential cross section as part of
rejection sampling.

.. doxygenclass:: celeritas::MuBremsDiffXsCalculator

Muon bremsstrahlung and pair production use a simple distribution to sample the
exiting polar angles.

.. doxygenclass:: celeritas::MuAngularDistribution

Photon scattering
-----------------

.. doxygenclass:: celeritas::KleinNishinaInteractor
.. doxygenclass:: celeritas::RayleighInteractor

Conversion/annihilation/photoelectric
-------------------------------------

.. doxygenclass:: celeritas::BetheHeitlerInteractor
.. doxygenclass:: celeritas::EPlusGGInteractor
.. doxygenclass:: celeritas::LivermorePEInteractor
.. doxygenclass:: celeritas::MuPairProductionInteractor

.. doxygenclass:: celeritas::AtomicRelaxation

Positron annihilation and Livermore photoelectric cross sections are calculated
on the fly (as opposed to pre-tabulated cross sections).

.. doxygenclass:: celeritas::EPlusGGMacroXsCalculator
.. doxygenclass:: celeritas::LivermorePEMicroXsCalculator

The energy transfer for muon pair production is sampled using the inverse
transform method with tabulated CDFs.

.. doxygenclass:: celeritas::MuPPEnergyDistribution

Coulomb scattering
------------------

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
and single scattering for large angles. The Wentzel VI model, used together
with the single Coulomb scattering model, is an implementation of the mixed
simulation algorithm. It is the default model in Geant4 above 100 MeV and
currently under development in Celeritas.

.. doxygenclass:: celeritas::CoulombScatteringInteractor
.. doxygenclass:: celeritas::WentzelDistribution
.. doxygenclass:: celeritas::WentzelHelper
.. doxygenclass:: celeritas::MottRatioCalculator

.. doxygenclass:: celeritas::ExpNuclearFormFactor
.. doxygenclass:: celeritas::GaussianNuclearFormFactor
.. doxygenclass:: celeritas::UUNuclearFormFactor

.. doxygenclass:: celeritas::detail::UrbanMscSafetyStepLimit
.. doxygenclass:: celeritas::detail::UrbanMscScatter

Discrete cross sections
=======================

Most physics processes use pre-calculated cross sections that are tabulated and
interpolated.

.. doxygenclass:: celeritas::XsCalculator

Cross sections for each process are evaluated at the beginning of the step
along with range limiters.

.. doxygenfunction:: celeritas::calc_physics_step_limit

If undergoing an interaction, the process is sampled from the stored
beginning-of-step cross sections.

.. doxygenfunction:: celeritas::select_discrete_interaction


Continuous slowing down
=======================

Most charged interactions emit one or more low-energy particles during their
interaction. Instead of creating explicit daughter tracks that are
immediately killed due to low energy, part of the interaction cross section is
lumped into a "slowing down" term that continuously deposits energy locally
over the step.

.. doxygenfunction:: celeritas::calc_mean_energy_loss

Since true energy loss is a stochastic function of many small collisions, the
*mean* energy loss term is an approximation. Additional
models are implemented to adjust the loss per step with stochastic sampling for
improved accuracy.

.. doxygenclass:: celeritas::EnergyLossHelper
.. doxygenclass:: celeritas::EnergyLossGammaDistribution
.. doxygenclass:: celeritas::EnergyLossGaussianDistribution
.. doxygenclass:: celeritas::EnergyLossUrbanDistribution

Imported data
=============

In addition to the core :ref:`api_importdata`, these import parameters are used
to provide cross sections, setup options, and other data to the EM physics.

.. doxygenstruct:: celeritas::ImportEmParameters
.. doxygenstruct:: celeritas::ImportAtomicTransition
.. doxygenstruct:: celeritas::ImportAtomicSubshell
.. doxygenstruct:: celeritas::ImportAtomicRelaxation

.. doxygenstruct:: celeritas::ImportLivermoreSubshell
.. doxygenstruct:: celeritas::ImportLivermorePE

.. doxygenstruct:: celeritas::ImportMuPairProductionTable
.. doxygentypedef:: celeritas::ImportSBTable
