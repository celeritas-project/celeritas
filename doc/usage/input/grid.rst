.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_grid:

Grids
=====

Tabulated physics data such as cross sections or energy loss are stored on
increasing, sorted 1D or 2D grids.

.. doxygenstruct:: celeritas::inp::Grid
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::UniformGrid
   :members:
   :no-link:

.. doxygenstruct:: celeritas::inp::TwodGrid
   :members:
   :no-link:

Both linear and spline interpolation are supported.

.. doxygenstruct:: celeritas::inp::Interpolation
   :members:
   :no-link:

