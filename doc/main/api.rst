.. Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api:

***********
Library API
***********

.. only:: nobreathe

   .. note:: The breathe_ extension was not used when building this version of
      the documentation. The API documentation will not be rendered below.

   .. _breathe: https://github.com/michaeljones/breathe#readme

The bulk of Celeritas' code is in several code libraries meant (with varying
degrees of polish) to be used by external users and application developers.
Currently, the most stable and user-ready component of Celeritas is its
:ref:`accel` code library for offloading to Geant4.

The Celeritas codebase lives under the ``src/`` directory and is partitioned
into several libraries of increasing complexity: ``corecel`` for GPU/CPU
abstractions, ``orange`` for a platform-portable geometry implementation,
``celeritas`` for the GPU implementation of physics and MC particle tracking,
and ``accel`` for the Geant4 integration library.

Additional top-level files provide access to version and
configuration attributes.

.. note::
   When building Celeritas, regardless of the configured :ref:`dependencies
   <Dependencies>`, all of the documented API code in ``corecel``, ``orange``,
   and ``celeritas`` (except possibly headers ending in ``.json.hh``,
   ``.device.hh``, etc.) will compile and can link to downstream code. However,
   some classes will throw ``celeritas::RuntimeError`` if they lack the required
   functionality.

   If Geant4 is disabled, the ``accel`` library will not be built or installed,
   because every component of that library requires Geant4.

.. toctree::
   api/corecel.rst
   api/orange.rst
   api/celeritas.rst
   api/accel.rst
