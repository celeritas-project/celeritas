.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_grids:

Numeric grids
=============

Structured data is used in cross section evaluation and sampling routines.

Grid access
-----------

Grids classes act like vectors, allowing direct access to node-centered data.

.. doxygenclass:: celeritas::UniformGrid
   :members:

.. doxygenclass:: celeritas::NonuniformGrid

These grids are often used with a helper function for safely finding points
that can be interpolated inside the grid's boundaries.

.. doxygenfunction:: celeritas::find_interp

Local interpolation
-------------------

At the core of between-point evaluation is interpolation, which can be done
as linear, logarithmic, semi-log, or spline.

.. doxygenclass:: celeritas::Interpolator
.. doxygenclass:: celeritas::SplineInterpolator

Grid calculation
----------------

.. doxygenclass:: celeritas::NonuniformGridCalculator
.. doxygenclass:: celeritas::UniformLogGridCalculator
.. doxygenclass:: celeritas::RangeCalculator
.. doxygenclass:: celeritas::SplineCalculator
.. doxygenclass:: celeritas::XsCalculator

Celeritas additionally supports two-dimensional interpolation, needed for some
sampling routines.

.. doxygenclass:: celeritas::TwodGridCalculator
.. doxygenclass:: celeritas::TwodSubgridCalculator

Inverse sampling
----------------

.. doxygenclass:: celeritas::InverseCdfFinder

Construction
------------

.. doxygenclass:: celeritas::SplineDerivCalculator
.. doxygenclass:: celeritas::RangeGridCalculator
