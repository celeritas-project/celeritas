.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0


.. _api_problem_def:

Problem definition
==================

Celeritas contains several high-level "parameter" classes that allow setup-time
access to problem data. These classes all correspond directly to "TrackView"
classes (see the `developer documentation`_ for details).

.. _developer documentation: https://celeritas-project.github.io/celeritas/dev/classes.html

.. doxygenclass:: celeritas::GeoParamsInterface

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams

.. doxygenclass:: celeritas::CutoffParams


.. _api_problem_setup:

Setting up problems
-------------------

Problem data is specified from applications and the Geant4 user interface using
the :ref:`input` API and loaded through "importers". Different front ends to
Celeritas use different sets of importers.

.. _api_problem_setup_standalone:

Standalone execution
^^^^^^^^^^^^^^^^^^^^

Standalone execution describes how to set up physics data (either through
Geant4 or loaded through an external file) and other problem properties.

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

.. doxygenstruct:: celeritas::inp::PointShape
.. doxygenstruct:: celeritas::inp::UniformBoxShape
.. doxygenstruct:: celeritas::inp::IsotropicAngle
.. doxygenstruct:: celeritas::inp::MonodirectionalAngle
.. doxygenstruct:: celeritas::inp::Monoenergetic

.. _api_problem_setup_framework:

User application/framework
^^^^^^^^^^^^^^^^^^^^^^^^^^

User applications define the system configuration, as well as what Celeritas
physics to enable (via :cpp:struct:`GeantImport`). Additional custom physics
can be added via the ``adjuster`` parameter to set or change any loaded data.

.. doxygenstruct:: celeritas::inp::FrameworkInput
   :members:
   :no-link:


Importers
^^^^^^^^^

Import options are read in to load problem input from various sources.

.. doxygenstruct:: celeritas::inp::FileImport
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::GeantImport
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::GeantDataImport
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::UpdateImport
   :members:
   :no-link:


Setup
^^^^^

.. doxygennamespace:: celeritas::setup

