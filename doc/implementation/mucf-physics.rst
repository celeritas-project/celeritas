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
after fusion occurred. For a general overview of the physics, refer to the
physics overview page.

.. toctree::
   :maxdepth: 2

   mucf-physics/physics-overview.rst

Input
-----

The input data is currently hardcoded in the
:cpp:class:`celeritas::inp::MucfPhysics` structure, which includes
temperature-dependent rates for mean cycle time, muonic atom transfer, and
muonic atom spin flip.

.. doxygenclass:: celeritas::inp::MucfPhysics

The muon-catalyzed fusion process is activated by enabling the ``mucf_physics``
option in the :cpp:class:`celeritas::ext::GeantPhysicsOptions` structure, and
for integration interfaces, the data is constructed if the
``G4MuonMinusAtomicCapture`` process is registered.
