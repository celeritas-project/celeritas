.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0


.. _api_problem_def:

Problem definition
==================

Celeritas contains several high-level "parameter" classes that allow setup-time
access to problem data. These classes all correspond directly to "TrackView"
classes (see the `developer documentation`_ for details).

.. _developer documentation: https://celeritas-project.github.io/celeritas/dev/classes.html

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams

.. doxygenclass:: celeritas::CutoffParams


.. _api_problem_setup:

Setting up problems
-------------------

Problem data is specified from applications and the Geant4 user interface using
the :ref:`input` API and loaded through "importers". Currently, Celeritas
relies on Geant4 and external Geant4 data files for its setup.

.. _api_problem_setup_standalone:

Standalone execution
^^^^^^^^^^^^^^^^^^^^

Standalone execution describes how to set up Geant4 physics and what events to
run.

.. doxygenstruct:: celeritas::inp::StandaloneInput

Standalone inputs must also specify the mechanism for loading primary
particles. The ``events`` field is a variant that can be one of these
structures:

.. doxygenstruct:: celeritas::inp::PrimaryGenerator
.. doxygenstruct:: celeritas::inp::SampleFileEvents
.. doxygenstruct:: celeritas::inp::ReadFileEvents

The primary generator, similar to Geant4's "particle gun", has different
configuration options:

.. doxygentypedef:: celeritas::inp::Events
.. doxygentypedef:: celeritas::inp::ShapeDistribution
.. doxygentypedef:: celeritas::inp::AngleDistribution
.. doxygentypedef:: celeritas::inp::EnergyDistribution

.. doxygenstruct:: celeritas::inp::PointDistribution
.. doxygenstruct:: celeritas::inp::UniformBoxDistribution
.. doxygenstruct:: celeritas::inp::IsotropicDistribution
.. doxygenstruct:: celeritas::inp::MonodirectionalDistribution
.. doxygenstruct:: celeritas::inp::MonoenergeticDistribution

.. _api_problem_setup_framework:

User application/framework
^^^^^^^^^^^^^^^^^^^^^^^^^^

User applications define the system configuration, as well as what Celeritas
physics to enable (via :cpp:struct:`GeantImport`). Additional custom physics
can be added via the ``adjuster`` parameter to set or change any loaded data.

.. doxygenstruct:: celeritas::inp::FrameworkInput
   :members:
   :no-link:


Loading data into Celeritas
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import options are read in to load problem input from various sources.

.. doxygenstruct:: celeritas::inp::PhysicsFromFile
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::PhysicsFromGeant
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::PhysicsFromGeantFiles
   :members:
   :no-link:


Setup
^^^^^

.. doxygennamespace:: celeritas::setup
