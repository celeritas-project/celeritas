.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Propagation and magnetic field
==============================

The propagation interface is built on top of the geometry to allow both curved
and straight-line movement. Field propagation is based on a composition of:

Field
  Maps a point in space and time to a field vector.
Equation of motion
  Calculates the path derivative of position and momentum given their current
  state and the templated field.
Integrator
  Numerically integrates a new position/momentum state given the start,
  path derivative, and step length.
Substepper
  Integrate a path segment that satisfies certain truncation error conditions,
  solving for the required segment length.
Propagator
  Given a maximum physics step, advance the geometry state and momentum along
  the field lines, satisfying constraints (see :ref:`field driver
  options<api_field_data>`) for the maximum geometry error.

.. _api_propagation:

Propagation
-----------

.. doxygenclass:: celeritas::LinearPropagator

.. doxygenclass:: celeritas::FieldPropagator

.. doxygenclass:: celeritas::FieldSubstepper

Field integration
-----------------

.. doxygenclass:: celeritas::DormandPrinceIntegrator

.. doxygenclass:: celeritas::RungeKuttaIntegrator

.. doxygenclass:: celeritas::ZHelixIntegrator

Magnetic field types
--------------------

.. doxygenclass:: celeritas::UniformField

.. doxygenclass:: celeritas::RZMapField

.. doxygenclass:: celeritas::CartMapField

.. doxygenclass:: celeritas::CylMapField

.. _api_field_data:

Field data input and options
----------------------------

JSON input for the field setup corresponds to the uniform field input
:cpp:struct:`celeritas::inp::UniformField` and the rz-map field input:

.. doxygenstruct:: celeritas::RZMapFieldInput
   :members:
   :no-link:

as well as fully Cartesian or cylindrical input:

.. doxygenstruct:: celeritas::CartMapFieldInput
   :members:
   :no-link:

.. doxygenstruct:: celeritas::CylMapFieldInput
   :members:
   :no-link:

The field driver options are not yet a stable part of the API:

.. doxygenstruct:: celeritas::FieldDriverOptions
   :members:
   :no-link:
