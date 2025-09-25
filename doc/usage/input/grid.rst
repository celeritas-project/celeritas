.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_grid:

Grids
=====

Tabulated physics data such as cross sections or energy loss are stored on
increasing, sorted 1D or 2D grids.

.. celerstruct:: inp::Grid
.. celerstruct:: inp::UniformGrid
.. celerstruct:: inp::TwodGrid

Both linear and spline interpolation are supported.

.. celerstruct:: inp::Interpolation
