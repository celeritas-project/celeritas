.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_em_physics:

**********
EM Physics
**********

The physics models in Celeritas are primarily derived from references cited by
Geant4, including the Geant4 physics reference manual. Undocumented adjustments
to those models in Geant4 may also be implemented.

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

.. toctree::
   :maxdepth: 2

   em-physics/summary.rst
   em-physics/slowing-down.rst
   em-physics/ionization.rst
   em-physics/brems.rst
   em-physics/coulomb.rst
   em-physics/photon-interaction.rst
   em-physics/pair-production.rst
   em-physics/annihilation.rst
