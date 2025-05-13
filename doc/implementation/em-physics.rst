.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_em_physics:

**********
EM Physics
**********

The physics models in Celeritas are primarily derived from references cited by
Geant4, including the Geant4 physics reference manual. Undocumented adjustments
to those models in Geant4 may also be implemented.

.. toctree::
   :maxdepth: 2

   em-physics/slowing-down.rst
   em-physics/ionization.rst
   em-physics/brems.rst
   em-physics/coulomb.rst
   em-physics/photon-interaction.rst
   em-physics/pair-production.rst
   em-physics/annihilation.rst


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
as properties of any secondary particles produced. Interaction limits are
described in :ref:`limits`.

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

Imported data
=============

Celeritas reads physics data from Geant4 (or from a ROOT file exported from
data previously loaded into Geant4). Different versions of Geant4 (and Geant4
data) can be used seamlessly with any version of Celeritas, allowing
differences to be isolated without respect to machine or model implementation.
