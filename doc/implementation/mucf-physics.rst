.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_mucf_physics:

************
MuCF Physics
************

The muon-catalyzed fusion physics in Celeritas is derived from custom
implementations written by Ara Knaian (Acceleron Fusion), Kevin Lynch
(Fermilab), and Sridhar Tripathy (UC Davis), not available in the standard
Geant4 source code.

Currently, the physics is managed by a single ``Executor`` that is responsible
for the full cycle, from atom formation to generating the outgoing secondaries
after fusion occurred.

Physics overview
================

Muons can be used to fuse deuterium-tritium mixtures at low temperatures
:cite:`kamimura-mucf-2023`. This is caused by the fact that molecular orbital
radii are inversely proportional to the mass of the lepton: the muon, with a
mass approximately 207 times larger than the electron's, leads to an orbital
radius about 207 times smaller. The reduced molecular orbital leads to a higher
nuclear wavefunction overlap, which in turn leads to a fusion reaction that does
not require high-temperature, magnetic-confined plasma to happen.

The full cycle time is a few orders of magnitude smaller than the average decay
time of the muon (:math:`2.2 \times 10^{-6}` s). Muonic atom formation takes
about :math:`10^{-12}-10^{-13}` s, muonic molecule formation takes
:math:`10^{-8}-10^{-10}` s, and the fusion process itself is at the order of
:math:`10^{-12}` s. In most instances, the muon is free after the fusion
process, leading to another cycle and giving the muon-catalyzed fusion its name.
The possible channels for all deuterium-tritium molecules are outlined below:

- :math:`(dd)_\mu`

  - :math:`\longrightarrow ^3\text{He} + \mu + n + 3.27 \ \text{MeV}`

  - :math:`\longrightarrow (^3\text{He})_\mu + n + 3.27 \ \text{MeV}`

  - :math:`\longrightarrow t + \mu + p + 4.03 \ \text{MeV}`

  - :math:`\longrightarrow (t)_\mu + p + 4.03 \ \text{MeV}`

- :math:`(dt)_\mu`

  - :math:`\longrightarrow \alpha + \mu + n + 17.6 \ \text{MeV}`

  - :math:`\longrightarrow (\alpha)_\mu + n + 17.6 \ \text{MeV}`

- :math:`(tt)_\mu`

  - :math:`\longrightarrow \alpha + \mu + 2n + 11.33 \ \text{MeV}`

  - :math:`\longrightarrow (\alpha)_\mu + 2n + 11.33 \ \text{MeV}`

In the cases where the muon sticks to an outgoing nucleus, e.g. generating a
:math:`(\alpha)_\mu`, the catalysis is halted. This happens at a fraction of a
percent to a few percent level, and the number that represents the fraction of
times this happens, with respect to the case where the muon is free, is called
the sticking factor.

A single muon can repeat this fusion cycle somewhat between 100 to 400 times.
The total number of fusion cycles produced by a single muon defines how much
energy can be extracted from it, in the effort of reaching a break-even
scenario. This is the threshold point where the energy required to generate the
muon is equal to the energy produced by said muon through the muCF cycles. The
sticking factor and the fusion cycle time are the main conditions that define
how many fusion cycles a muon can undergo. The fusion cycle time depends on the
d-t mixture, its temperature, and on the final spin of the molecule. Only muonic
molecules where the total spin :math:`F = I_N \pm 1/2` is on, or has a
projection onto the total angular momentum J = 1 are reactive. The spin states
of the three possible muonic molecules are summarized in table
:numref:`muon_spin_states`.


.. _muon_spin_states:

.. table:: Spin states of dt muonic molecules

   +------------------+-----------+--------------+-------------------------+---------------------+
   | Molecule         | Nuclei    |  :math:`I_N` | :math:`F = I_N \pm 1/2` | Reactive states (F) |
   +==================+===========+==============+=========================+=====================+
   | :math:`(dd)_\mu` | 1, 1      | 0, 1, 2      | 1/2, 3/2, 5/2           | 1/2, 3/2            |
   +------------------+-----------+--------------+-------------------------+---------------------+
   | :math:`(dt)_\mu` | 1, 1/2    | 1/2, 3/2     | 0, 1, 2                 | 0, 1                |
   +------------------+-----------+--------------+-------------------------+---------------------+
   | :math:`(tt)_\mu` | 1/2, 1/2  | 0, 1         | 1/2                     | 1/2                 |
   +------------------+-----------+--------------+-------------------------+---------------------+

Input
=====

The input data is currently hardcoded in the
:cpp:class:`celeritas::inp::MucfPhysics` structure, which includes
temperature-dependent rates for mean cycle time, muonic atom transfer, and
muonic atom spin flip. The muon-catalyzed fusion process is activated by
enabling the ``mucf_physics`` option in
:cpp:class:`celeritas::ext::GeantPhysicsOptions`.

.. celerstruct:: inp::MucfPhysics

.. todo:: Expand description when hardcoded data is finalized.

Geant4 integration
------------------

For integration interfaces, if ``mucf_physics`` option in
:cpp:class:`celeritas::ext::GeantPhysicsOptions` is enabled, the muon-catalyzed
fusion data is constructed when the ``G4MuonMinusAtomicCapture`` process is
registered.

.. todo:: Add process/model/executor details
