.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _optical_generating:

Generating photons
==================

Photon generation in Celeritas is split into multiple steps to allow extension
and to improve performance on GPUs by minimizing memory usage.
:numref:`fig-optical-gen-flow` depicts the creation of optical photons in the
optical tracking loop as a flowchart.

.. _fig-optical-gen-flow:

.. figure:: /_static/dot/optical-gen-flow.*
   :align: center
   :width: 80%

   Creation of optical photons in the optical tracking loop.


During the main :ref:`stepping loop <api_stepping>`, the
:cpp:class:`celeritas::OpticalCollector`
class adds a pre-step hook to store each track's speed, position, time, and
material; at the end of the step, the track's updated properties and
within-step energy distribution are used to "offload" optical photons by
generating *distribution parameters* to be sampled in the stepping loop.

.. doxygenclass:: celeritas::OpticalCollector
.. doxygenclass:: celeritas::CherenkovOffload
.. doxygenclass:: celeritas::ScintillationOffload

Distributions
-------------

The *generator distribution* data for cherenkov and scintillation photons is
analogous to the "genstep" data structure in Opticks
:cite:`blyth-opticks-2019`.

.. celerstruct:: optical::GeneratorDistributionData

For testing and independent execution, the user can also manually define
distributions via the input.

Generators
----------

Generator distributions specify the distribution parameters for different
model-dependent Generators.

.. doxygenclass:: celeritas::optical::CherenkovGenerator
.. doxygenclass:: celeritas::optical::ScintillationGenerator
.. doxygenclass:: celeritas::optical::WavelengthShiftGenerator
.. doxygenclass:: celeritas::optical::PrimaryGenerator

Each of these class defines a "call" operator that takes a random number engine
to produce an optical photon.

Initialization
--------------

The generators above are sampled inside a kernel, producing one or more "track
initializers" that are the starting state for an optical photon:

.. celerstruct:: optical::TrackInitializer

Cherenkov and scintillation photons are sampled using the GeneratorAction to
fill empty optical track slots.

.. doxygenclass:: celeritas::optical::GeneratorAction

The PrimaryGeneratorAction class interfaces with the user (for testing or
direct offloading optical photons from Geant4) to create tracks directly from
initializers:

.. doxygenclass:: celeritas::optical::PrimaryGeneratorAction
