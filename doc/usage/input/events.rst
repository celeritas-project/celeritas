.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_events:

Events
======

Standalone inputs must specify how the primary particles are generated. These
events can be generated internally or read from an external file.

Core events
-----------

.. celerstruct:: inp::Events

The ``generator`` field in :cpp:struct:`Events` selects one of the supported
event sources.

.. doxygentypedef:: celeritas::inp::Generator

.. celerstruct:: inp::PrimaryGenerator
.. celerstruct:: inp::SampleFileEvents
.. celerstruct:: inp::ReadFileEvents

Optical events
--------------

Optical photons can be generated through several different mechanisms.

.. doxygentypedef:: celeritas::inp::OpticalGenerator

.. celerstruct:: inp::OpticalPrimaryGenerator
.. celerstruct:: inp::OpticalEmGenerator
.. celerstruct:: inp::OpticalOffloadGenerator
.. celerstruct:: inp::OpticalDirectGenerator

The standalone optical application ``celer-optical`` currently supports photon
generation though a primary generator or from offloaded distribution data read
from a file.

Primary generators
------------------

The ``CorePrimaryGenerator`` and ``OpticalPrimaryGenerator`` are analogous to
Geant4's particle gun and provide distributions for sampling the initial
particle direction, energy, and position.

.. doxygentypedef:: celeritas::inp::AngleDistribution
.. doxygentypedef:: celeritas::inp::EnergyDistribution
.. doxygentypedef:: celeritas::inp::ShapeDistribution

The fixed-value delta distributions for angle, energy, and position are aliased
by the following types:

.. doxygentypedef:: celeritas::inp::MonodirectionalDistribution
.. doxygentypedef:: celeritas::inp::MonoenergeticDistribution
.. doxygentypedef:: celeritas::inp::PointDistribution

Distributions
-------------

The following distribution types can be used for sampling values. A
distribution can optionally be truncated to a specified interval, while a delta
distribution represents a fixed value rather than a sampled quantity.

.. celerstruct:: inp::TruncatedDistribution
.. celerstruct:: inp::DeltaDistribution
.. celerstruct:: inp::NormalDistribution
.. celerstruct:: inp::UniformBoxDistribution
.. celerstruct:: inp::IsotropicDistribution
