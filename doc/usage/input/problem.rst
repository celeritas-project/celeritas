.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_problem:

Problem setup
=============

Virtually all attributes of the problem and its execution are defined through
the Problem input class, which is set up in part by the user and in part by
external applications as directed by the user.

.. celerstruct:: inp::Problem

.. _inp_framework:

User application/framework
--------------------------

User applications define the system configuration, as well as what Celeritas
physics to enable (via :cpp:struct:`GeantImport`). Additional custom physics
can be added via the ``adjust`` member to set or change any loaded data.

.. celerstruct:: inp::FrameworkInput

Loading data into Celeritas
---------------------------

Import options are read in to load problem input from Geant4 and other sources.

.. celerstruct:: inp::PhysicsFromFile
.. celerstruct:: inp::PhysicsFromGeant
.. celerstruct:: inp::PhysicsFromGeantFiles

.. _inp_standalone_input:

Standalone execution
--------------------

Standalone execution describes how to set up Geant4 physics and what events to
run.

.. celerstruct:: inp::StandaloneInput

Standalone inputs must also specify the mechanism for loading primary
particles. The ``events`` field is a variant that can be one of these
structures:

.. celerstruct:: inp::PrimaryGenerator
.. celerstruct:: inp::SampleFileEvents
.. celerstruct:: inp::ReadFileEvents

The primary generator, similar to Geant4's "particle gun", has different
configuration options:

.. doxygentypedef:: celeritas::inp::Events
.. doxygentypedef:: celeritas::inp::ShapeDistribution
.. doxygentypedef:: celeritas::inp::AngleDistribution
.. doxygentypedef:: celeritas::inp::EnergyDistribution

.. celerstruct:: inp::PointDistribution
.. celerstruct:: inp::UniformBoxDistribution
.. celerstruct:: inp::IsotropicDistribution
.. celerstruct:: inp::MonodirectionalDistribution
.. celerstruct:: inp::MonoenergeticDistribution


.. _inp_system:

System
------

Some low-level system options, such as enabling GPU, are set up once per program
execution. They are not loaded by the :cpp:struct:`Problem` definition but
are used by the standalone/framework inputs.

.. celerstruct:: inp::System
.. celerstruct:: inp::Device
