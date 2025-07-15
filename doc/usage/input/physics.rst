.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_physics:

Physics
=======

.. doxygenstruct:: celeritas::inp::Physics
   :members:
   :no-link:

Electromagnetic
^^^^^^^^^^^^^^^

.. doxygenstruct:: celeritas::inp::EmPhysics
   :members:
   :no-link:

Decay
^^^^^

.. doxygenstruct:: celeritas::inp::DecayPhysics
   :members:
   :no-link:

Hadronic
^^^^^^^^

.. doxygenstruct:: celeritas::inp::HadronicPhysics
   :members:
   :no-link:

Optical
^^^^^^^

Optical photon _generation_ is a part of the standard stepping loop that manages
EM, decay, and hadronic physics, but its _transport_ has its own separate
stepping loop, where surface physics is the most complex part. Therefore, the
``OpticalPhysics`` input includes optical photon _generation_ processes (such as
Cherenkov and scintillation) and surface physics information. The latter
describing how optical photons should interact with it.

.. doxygenstruct:: celeritas::inp::OpticalPhysics
   :members:
   :no-link:

``SurfacePhysics`` (below) untangles the Geant4 surface model design, which
leads to high code-branching, into separate mechanisms for surface reflectivity,
roughness, and interaction models. In the Celeritas design, a given Geant4 model
(e.g. GLISUR or Unified) is represented by a combination of those 3 mechanisms.
This unfolding leads to a less simple input, but reduces kernel size/complexity
on the GPU. 

.. doxygenstruct:: celeritas::inp::SurfacePhysics
    :members:
    :no-link:


Processes
---------

.. doxygenstruct:: celeritas::inp::BremsProcess
   :members:
   :no-link:


Models
------

.. doxygenstruct:: celeritas::inp::SeltzerBergerModel
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::RelBremsModel
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::MuBremsModel
   :members:
   :no-link:
