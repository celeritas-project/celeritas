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

.. todo:: Add description of new decay process class and its inclusion in the stepping loop when implemented

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

.. doxygenclass:: celeritas::MuDecayInteractor
