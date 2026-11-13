.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_decay_physics:

*************
Decay physics
*************

Similar to EM physics, the particle decay processes in Celeritas are implemented
following Geant4's reference manual and source code.

Decay processes
===============

A single :cpp:class:`celeritas::DecayProcess` manages all supported decay
channels. Each channel (derived from :cpp:class:`celeritas::DecayChannel`) is
the step action that samples the decay products, analogous to a model of an
interaction process. In flight, decay competes with the particle's
interactions through a "cross section" equal to the inverse of its mean decay
length. At rest, decay applies only if the particle has no other at-rest
process: for example, a stopped :math:`\mu^-` undergoes muon-catalyzed fusion
instead of decay when :ref:`muCF physics <api_mucf_physics>` is enabled. When
decay at rest is selected, the track time is advanced by a remaining lifetime
sampled from the particle's decay constant.

.. doxygenclass:: celeritas::DecayProcess
.. doxygenclass:: celeritas::DecayChannel

The table lists all available particles that undergo any decay type, along with
their decay channels branching ratios, and associated Interactor.

.. only:: html

   .. table:: Particle decays available in Celeritas.

      +----------------+---------------------------------+----------------------------------------+-------------------------------------------+
      | Particle       | Decay channel(s)                | Fraction (:math:`\Gamma_i/\Gamma`)     | Celeritas Implementation                  |
      +================+=================================+========================================+===========================================+
      | :math:`\mu^-`  | :math:`e^- \bar{\nu}_e \nu_\mu` | 1                                      | :cpp:class:`celeritas::MuDecayInteractor` |
      +----------------+---------------------------------+----------------------------------------+                                           |
      | :math:`\mu^+`  | :math:`e^+ \nu_e \bar{\nu}_\mu` | 1                                      |                                           |
      +----------------+---------------------------------+----------------------------------------+-------------------------------------------+

.. only:: latex

   .. raw:: latex

      \begin{table}[h]
          \caption{Particle decays available in Celeritas.}
          \begin{threeparttable}
          \begin{tabular}{llll}
            \toprule
            Particle & Decay channel(s) & Fraction ($\Gamma_i/\Gamma$) & Celeritas Implementation \\
            \midrule
            $\mu^-$           & $e^- \bar{\nu}_e \nu_\mu$ & 1                                     & \multirow{2}{*}{\texttt{\scriptsize celeritas::MuDecayInteractor}} \\
            \cline{1-3}
            $\mu^+$           & $e^+ \nu_e \bar{\nu}_\mu$ & 1                                     & \\
            \bottomrule
          \end{tabular}
          \end{threeparttable}
      \end{table}

Muon decay
----------

.. doxygenclass:: celeritas::MuDecayChannel
.. doxygenclass:: celeritas::MuDecayInteractor

Input
=====

Decay is described by :cpp:class:`celeritas::inp::DecayPhysics`, which lists
the supported decay channels and the particles each applies to.

.. celerstruct:: inp::DecayPhysics
.. celerstruct:: inp::MuonDecay

Geant4 integration
------------------

Decay is imported for particles that have the ``G4Decay`` process. Muon decay
is enabled for :math:`\mu^\pm` whose decay table contains only
``G4MuonDecayChannel``. Decay of other particles, or through other channels, is
ignored with a warning. With the Celeritas EM physics list, enabling the
``decay`` option of :cpp:class:`celeritas::GeantMuonPhysicsOptions` adds
:cpp:class:`celeritas::SupportedDecayPhysics`, which registers ``G4Decay`` for
muons only. The FTFP_BERT physics list always includes decay through
``G4DecayPhysics``, so the option is not used there. User physics lists can
add either one, but not both.
