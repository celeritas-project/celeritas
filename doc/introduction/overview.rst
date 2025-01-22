.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Overview
========

This user manual is written for three audiences with different goals: Geant4
toolkit users for integrating Celeritas as a plugin, advanced users for
extending Celeritas with new physics, and developers for maintaining and
advancing the codebase.

Installation and usage
----------------------

The :ref:`infrastructure` section describes how to obtain and set up a working
copy of Celeritas. Once installed, :ref:`Celeritas can be used
<infrastructure>` as a software library for integrating directly into
experiment frameworks and user applications, or its front end applications can
be used to evaluate performance benchmarks and perform some simple analyses.

GPU usage
---------

Celeritas automatically copies data to device when constructing objects as long
as the GPU is enabled. See :ref:`api_system` for details on initializing and
accessing the device.

Geometry
--------

Celeritas has two choices of geometry implementation. VecGeom_ is a
CUDA-compatible library for navigation on Geant4 detector geometries.
:ref:`api_orange` is a work in progress for surface-based geometry navigation
that is "platform portable", i.e. able to run on GPUs from multiple vendors.

Celeritas wraps both geometry packages with a uniform interface for changing
and querying the geometry state.

.. _VecGeom: https://gitlab.cern.ch/VecGeom/VecGeom

Units
-----

The Celeritas default unit system is Gaussian CGS_, but it can be
:ref:`configured <configuration>` to use SI or CLHEP unit systems as well. A
compile-time metadata class allows simultaneous use of macroscopic-scale units
and atomic-scale values such as MeV. For more details, see the
:ref:`units_constants` section of the API documentation.

.. _CGS: https://en.wikipedia.org/wiki/Gaussian_units

EM Physics
----------

Celeritas implements physics processes and models for transporting electron,
positron, and gamma particles. Initial support is being added for muon EM
physics.  Implementation details of these models and their corresponding Geant4
classes are documented in :ref:`api_em_physics`.

Optical Physics
---------------

Optical physics are being added to Celeritas to support various high energy
physics and nuclear physics experiments including LZ, Calvision, DUNE, and
ePIC. See the :ref:`api_optical_physics` section of the implementation details.

Stepping loop
-------------

In Celeritas, the core algorithm is a loop interchange between particle
tracks and steps. Traditionally,
in a CPU-based simulation, the outer loop iterates over particle tracks, while
the inner loop handles steps. Each step includes actions such as evaluating cross sections,
calculating distances to geometry boundaries, and managing interactions that
produce secondaries.

Celeritas vectorizes this process by reversing the loop structure on the GPU.
The outer loop is over *step iterations*, and the inner loop processes *track
slots*, which are elements in a fixed-size vector of active tracks. The
stepping loop in Celeritas is thus a sorted loop over *actions*, with each
action typically corresponding to a kernel launch on the GPU (or an inner loop
over tracks when running on the CPU).

See :ref:`api_stepping` for implementation details on the ordering of actions
and the status of a track slot during iteration.

