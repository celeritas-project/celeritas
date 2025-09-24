.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_physics:

Physics
=======

.. note:: For discussion of model and process naming, see
   issue 1562_ .

.. _1562: https://github.com/celeritas-project/celeritas/pull/1562

The selection and data used to create physics classes will be defined
underneath this struct.

.. doxygenstruct:: celeritas::inp::Physics
   :members:
   :no-link:

Electromagnetic
^^^^^^^^^^^^^^^

.. doxygenstruct:: celeritas::inp::EmPhysics
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

Celeritas' ``SurfacePhysics`` implementation is designed differently from Geant4
and is meant to reduce code branching. In Geant4 one single model (e.g. Unified)
encompasses interface type (e.g. dielectric-dielectric), multiple reflection
mechanisms (specular spike, specular lobe, etc.), and surface roughness types
(polished or Gaussian). Celeritas describes an optical surface based on its
interface type, the possible reflection mechanisms it can undergo, its
roughness, and other parameters, such as detector efficiency in case it is a
scoring surface for an optical detector such as a PMT or SiPM. A specific Geant4
model (e.g. Unified) is simply a specific combination of such characteristics.
This "model unfolding" leads to a less simple input definition compared to
Geant4, but allows for a more general surface definition system. This leads to
better extensibility and reduces kernel size/complexity on the GPU.

.. doxygenstruct:: celeritas::inp::SurfacePhysics
    :members:
    :no-link:


Processes
---------

.. doxygenstruct:: celeritas::inp::BremsstrahlungProcess
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::PairProductionProcess
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::PhotoelectricProcess
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::AtomicRelaxation
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

.. doxygenstruct:: celeritas::inp::BetheHeitlerProductionModel
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::MuPairProductionModel
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::LivermorePhotoModel
   :members:
   :no-link:
