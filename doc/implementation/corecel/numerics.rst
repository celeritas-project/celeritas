.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Numerics
--------

These functions replace and extend those in the C++ standard library
``<cmath>``, ``<numeric>``, and ``<atomic>`` headers.

Math
^^^^

Direct replacements for ``<cmath>`` and ``<limits>``:

.. doxygenfunction:: celeritas::fma
.. doxygenstruct:: celeritas::numeric_limits
   :members:

Portable replacements for math functions provided by the CUDA/HIP libraries:

.. doxygenfunction:: celeritas::popcount
.. doxygenfunction:: celeritas::rsqrt(double)
.. doxygenfunction:: celeritas::sincos(double a, double* s, double* c)
.. doxygenfunction:: celeritas::sincospi(double a, double* s, double* c)

Additional utility functions:

.. doxygenfunction:: celeritas::fastpow
.. doxygenfunction:: celeritas::ceil_div
.. doxygenfunction:: celeritas::clamp_to_nonneg
.. doxygenfunction:: celeritas::eumod
.. doxygenfunction:: celeritas::ipow
.. doxygenfunction:: celeritas::negate
.. doxygenfunction:: celeritas::signum

Several standalone classes can be used to evaluate, integrate, and solver
expressions and functions.

.. doxygenclass:: celeritas::PolyEvaluator
.. doxygenclass:: celeritas::Integrator
.. doxygenclass:: celeritas::BisectionRootFinder
.. doxygenclass:: celeritas::IllinoisRootFinder
.. doxygenclass:: celeritas::TridiagonalSolver
.. doxygenclass:: celeritas::FerrariSolver

Atomics
^^^^^^^

These atomic functions are for use in kernel code (CUDA/HIP/OpenMP) that use
track-level parallelism.

.. doxygenfunction:: celeritas::atomic_add
.. doxygenfunction:: celeritas::atomic_min
.. doxygenfunction:: celeritas::atomic_max

Array utilities
^^^^^^^^^^^^^^^

These operate on fixed-size arrays of data (see :ref:`api_containers`), usually
``Real3`` as a Cartesian spatial coordinate.

.. doxygentypedef:: celeritas::Real3

.. doxygenfunction:: celeritas::axpy
.. doxygenfunction:: celeritas::dot_product
.. doxygenfunction:: celeritas::cross_product
.. doxygenfunction:: celeritas::norm(Array<T, N> const &v)
.. doxygenfunction:: celeritas::make_orthogonal
.. doxygenfunction:: celeritas::make_unit_vector
.. doxygenfunction:: celeritas::distance
.. doxygenfunction:: celeritas::from_spherical
.. doxygenfunction:: celeritas::rotate

Soft equivalence
^^^^^^^^^^^^^^^^

These utilities are used for comparing real-valued numbers to a given
tolerance.

.. doxygenclass:: celeritas::SoftEqual
.. doxygenclass:: celeritas::SoftZero
.. doxygenclass:: celeritas::EqualOr
.. doxygenclass:: celeritas::ArraySoftUnit

.. doxygenfunction:: celeritas::is_soft_orthogonal
