.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: none

.. _api_orange:

ORANGE
======

The Oak Ridge Advanced Nested Geometry Engine (ORANGE)
:cite:`orange-tm` is a surface-based geometry that has been adapted to GPU
execution to support platform portability in Celeritas. It can be built via its
interface to SCALE or constructed automatically from Geant4 geometry
representation.

.. doxygenclass:: celeritas::OrangeParams

.. toctree::
   :maxdepth: 2

   orange/translation.rst
   orange/construction.rst
   orange/runtime.rst
