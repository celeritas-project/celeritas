.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _overview:

********
Overview
********

High Energy Physics (HEP) is entering an exciting era for potential scientific
discovery. There is now overwhelming evidence that the Standard Model (SM) of
particle physics is incomplete. A targeted program, as recommended by the
Particle Physics Project Prioritization Panel (P5), has been designed to reveal
the nature and origin of the physics Beyond Standard Model (BSM). Two of the
flagship projects are the upcoming high luminosity upgrade of the High
Luminosity Large Hadron Collider (HL-LHC) and its four main detectors and the
Deep Underground Neutrino Experiment (DUNE) at the Sanford Underground Research
Facility (SURF) and Fermi National Accelerator Laboratory (Fermilab). Only by
comparing these detector results to detailed Monte Carlo (MC) simulations can
new physics be discovered. The quantity of simulated MC data must be many times
that of the experimental data to reduce the influence of statistical effects
and to study the detector response over a very large phase space of new
phenomena. Additionally, the increased complexity, granularity, and readout
rate of the detectors require the most accurate, and thus most compute
intensive, physics models available. However, projections of the computing
capacity available in the coming decade fall far short of the estimated
capacity needed to fully analyze the data from the HL-LHC. The contribution to
this estimate from MC full detector simulation is based on the performance of
the current state-of-the-art and LHC baseline MC application Geant4, a threaded
CPU-only code whose performance has stagnated with the deceleration of clock
rates and core counts in conventional processors.

General-purpose accelerators offer far higher performance per watt than Central
Processing Units (CPUs). Graphics Processing Units (GPUs) are the most common
such devices and have become commodity hardware at the U.S. Department of
Energy (DOE) Leadership Computing Facilities (LCFs) and other
institutional-scale computing clusters. However, adapting scientific codes to
run effectively on GPU hardware is nontrivial and results both from core
algorithmic properties of the physics and from implementation choices over the
history of an existing scientific code. The high sensitivity of GPUs to memory
access patterns, thread divergence, and device occupancy makes effective
adaptation of MC physics algorithms especially challenging.

Our objective is to advance and mature the new GPU-optimized code Celeritas  [#celeritas_vers]_  to
run full-fidelity MC simulations of LHC detectors. The primary goal of the
Celeritas project is to maximize utilization of HEP computing facilities and
the DOE LCFs to extract the ever-so-subtle signs of new physics. It aims to
reduce the computational demand of the HL-LHC to meet the available supply,
using the advanced architectures that will form the backbone of high
performance computing (HPC) over the next decade. Enabling HEP science at the
HL-LHC will require MC detector simulation that executes the latest and best
physics models and achieves high performance on accelerated hardware.

.. [#celeritas_vers] This documentation is generated from Celeritas |release|.
