.. Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
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
Driver
  Integrate path segments that satisfy certain error conditions, solving for
  the required segment length.
Propagator
  Given a maximum physics step, advance the geometry state and momentum along
  the field lines, satisfying constraints (see :ref:`field driver
  options<api_field_data>`) for the maximum geometry error.

Propagation
-----------

.. doxygenclass:: celeritas::LinearPropagator

.. doxygenclass:: celeritas::FieldPropagator

.. doxygenfunction:: celeritas::make_mag_field_propagator


.. _api_field_data:

Field data input and options
----------------------------

These classes correspond to JSON input files to the field setup.

.. doxygenstruct:: celeritas::UniformFieldParams
   :members:
   :no-link:

.. doxygenstruct:: celeritas::RZMapFieldInput
   :members:
   :no-link:


The field driver options are not yet a stable part of the API:

.. doxygenstruct:: celeritas::FieldDriverOptions
   :members:
   :no-link:

