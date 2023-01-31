.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _introduction:

************
Introduction
************

New projects in High Energy Physics (HEP) and upgrades to existing ones promise
new discoveries but at the cost of increased hardware complexity and data
readout rates. Deducing new physics from detector readouts requires a
proportional increase in computational resources. The High Luminosity Large
Hadron Collider (HL-LHC) detectors will require more computational resources
than are available with traditional CPU-based computing grids. For example, the
CMS collaboration forecasts :cite:`2021-CMS-Offline` that when the upgrade is
brought online, computational resource requirements will exceed availability by
more than a factor of two, about 40% of which is Monte Carlo (MC) detector
simulation, without substantial research and development improvements.

Celeritas [#celeritas_vers]_ is a new MC particle transport code designed for
high performance simulation of complex HEP detectors on GPU-accelerated
hardware.  Its immediate goal is to simulate electromagnetic (EM) physics for
LHC-HL detectors with no loss in fidelity, acting as a plugin to accelerate
existing Geant4 :cite:`Geant4` workflows by "offloading" selected particles to
Celeritas to transport on GPU.

This user manual is written for three audiences with different goals: Geant4
toolkit users for integrating Celeritas as a plugin, advanced users for
extending Celeritas with new physics, and developers for maintaining and
advancing the codebase.  The code is partitioned into several libraries
of increasing complexity: ``corecel`` for GPU/CPU abstractions, ``orange`` for
a platform-portable geometry implementation, ``celeritas`` for the GPU
implementation of physics and MC particle tracking, and ``accel`` for the
Geant4 integration library.

.. [#celeritas_vers] This documentation is generated from Celeritas |release|.
