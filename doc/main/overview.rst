.. Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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

EM Physics
==========

Celeritas implements physics processes and models for transporting electron,
positron, and gamma particles as shown in the accompanying table. Initial
support is being added for muon physics and is not shown below.
Implementation details of these models
and their corresponding Geant4 classes are documented in :ref:`api_em_physics`.

.. only:: html

   .. table:: Electromagnetic physics processes and models available in Celeritas.

      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | **Particle**   | **Processes**       |  **Models**                 | **Celeritas Implementation**                        | **Applicability**        |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
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
      | :math:`\mu^-`  | Ionization          |  ICRU73QO                   | :cpp:class:`celeritas::BraggICRU73QOInteractor`     |       0--200 keV         |
      |                +                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Bethe--Bloch               | :cpp:class:`celeritas::MuBetheBlochInteractor`      |   200 keV--100 TeV       |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Bremsstrahlung      |  Mu bremsstrahlung          | :cpp:class:`celeritas::MuBremsstrahlungInteractor`  |       0--100 TeV         |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      | :math:`\mu^+`  | Ionization          |  Bragg                      | :cpp:class:`celeritas::BraggICRU73QOInteractor`     |       0--200 keV         |
      |                +                     +-----------------------------+-----------------------------------------------------+--------------------------+
      |                |                     |  Bethe--Bloch               | :cpp:class:`celeritas::MuBetheBlochInteractor`      |   200 keV--100 TeV       |
      |                +---------------------+-----------------------------+-----------------------------------------------------+--------------------------+
      |                | Bremsstrahlung      |  Mu bremsstrahlung          | :cpp:class:`celeritas::MuBremsstrahlungInteractor`  |       0--100 TeV         |
      +----------------+---------------------+-----------------------------+-----------------------------------------------------+--------------------------+

.. only:: latex

   .. raw:: latex

      \begin{table}[h]
        \caption{Electromagnetic physics processes and models available in Celeritas.}
        \begin{threeparttable}
        \begin{tabular}{| l | l | l | l | r | }
          \hline
          \textbf{Particle}         & \textbf{Processes}                  & \textbf{Models}      & \textbf{Celeritas Implementation}                           & \textbf{Applicability} \\
          \hline
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
          \multirow{3}{*}{$\mu^-$}  & \multirow{2}{*}{Ionization}         & ICRU73QO             & \texttt{\scriptsize celeritas::BraggICRU73QOInteractor}     & 0--200 keV \\
                                                                          \cline{3-5}
                                    &                                     & Bethe--Bloch         & \texttt{\scriptsize celeritas::MuBetheBlochInteractor}      & 200 keV -- 100 TeV \\
                                    \cline{2-5}
                                    & Bremsstrahlung                      & Mu bremsstrahlung    & \texttt{\scriptsize celeritas::MuBremsstrahlungInteractor}  & 0--100 TeV \\
          \hline
          \multirow{3}{*}{$\mu^+$}  & \multirow{2}{*}{Ionization}         & Bragg                & \texttt{\scriptsize celeritas::BraggICRU73QOInteractor}     & 0--200 keV \\
                                                                          \cline{3-5}
                                    &                                     & Bethe--Bloch         & \texttt{\scriptsize celeritas::MuBetheBlochInteractor}      & 200 keV -- 100 TeV \\
                                    \cline{2-5}
                                    & Bremsstrahlung                      & Mu bremsstrahlung    & \texttt{\scriptsize celeritas::MuBremsstrahlungInteractor}  & 0--100 TeV \\
          \hline
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

Coulomb scattering
------------------

Elastic scattering of charged particles can be simulated in three ways:

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

Optical Physics
===============

TODO:

- Describe integration into the main stepping loop
- Add mermaid plot of optical stepping loop
- Describe pre-generation, generation
- Add optical models

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

The core algorithm in Celeritas is to perform a *loop interchange*
:cite:`allen_automatic_1984` between particle tracks and steps. The classical
(serial) way of simulating an event is to have an outer loop over tracks and an
inner loop over steps, and inside each step are the various actions applied to
a track such as evaluating cross sections, calculating the distance to the
nearest geometry boundary, and undergoing an interaction to produce
secondaries. In Python pseudocode this looks like:

.. code-block:: python

   track_queue = primaries
   while track_queue:
      track = track_queue.pop()
      while track.alive:
         for apply_action in [pre, along, post]:
            apply_action(track)
         track_queue += track.secondaries

There is effectively a data dependency between the track at step *i* and step
*i + 1* that prevents vectorization. The approach Celeritas takes to
"vectorize" the stepping loop on GPU is to have an outer loop over "step
iterations" and an inner loop over "track slots", which are elements in a
fixed-size vector of tracks that may be in flight:

.. code-block:: python

   initializers = primaries
   track_slots = [None] * num_track_slots
   while initializers or any(track_slots):
      fill_track_slots(track_slots, initializers)
      for apply_action in [pre, along, post]:
         for (i, track) in enumerate(track_slots):
            apply_action(track)
            track_queue += track.secondaries
      if not track.alive:
         track_slots[i] = None


The stepping loop in Celeritas is therefore a sorted loop over "actions", each
of which is usually a kernel launch (or an inner loop over tracks if running on
CPU).

See :ref:`api_stepping` for implementation details on the ordering of actions
and the status of a track slot during iteration.

GPU usage
=========

Celeritas automatically copies data to device when constructing objects as long
as the GPU is enabled.
