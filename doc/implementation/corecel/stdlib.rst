.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Standard library replacements
-----------------------------

These functions replace or extend those in the C++ standard library but work in
GPU code without the special ``--expt-relaxed-constexpr`` flag. None of these
algorithms are thread-cooperative, so all can be used in standard Celeritas
kernels.

Utilities
^^^^^^^^^

A few utilities replace and supplement the standard ``<utility>`` header.

.. doxygenfunction:: celeritas::forward
.. doxygenfunction:: celeritas::move
.. doxygenfunction:: celeritas::trivial_swap
.. doxygenfunction:: celeritas::exchange

Algorithms
^^^^^^^^^^

These device-compatible functions replace or extend those in the C++ standard
library ``<algorithm>`` header. The implementations of ``sort`` and other
partitioning elements are derived from LLVM's ``libc++``.

.. doxygenfunction:: celeritas::all_of
.. doxygenfunction:: celeritas::any_of
.. doxygenfunction:: celeritas::clamp
.. doxygenfunction:: celeritas::lower_bound
.. doxygenfunction:: celeritas::max
.. doxygenfunction:: celeritas::min
.. doxygenfunction:: celeritas::min_element
.. doxygenfunction:: celeritas::partition
.. doxygenfunction:: celeritas::sort
.. doxygenfunction:: celeritas::upper_bound

A few convenience algorithms are built on top of these replacements:

.. doxygenfunction:: celeritas::all_adjacent
.. doxygenfunction:: celeritas::lower_bound_linear
.. doxygenfunction:: celeritas::find_sorted
