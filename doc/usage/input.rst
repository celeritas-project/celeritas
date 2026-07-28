.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _input:

*****
Input
*****

.. note:: This section documents the input interface planned for Celeritas 1.0.
   It is currently available only through the standalone applications
   ``celer-sim`` and ``celer-optical``.

.. only:: nobreathe

   .. warning:: The breathe_ extension was not used when building this version
      of the documentation. The input definitions will not be displayed.

   .. _breathe: https://github.com/michaeljones/breathe#readme

The standalone applications and, beginning with Celeritas 1.0, the library
interface for external integration use a single interface to define properties
about the simulation to be run. This interface is a nested set of simple struct
objects that are used both to enable options and to set up low-level C++ data
structures. Many of the struct names in the ``inp`` namespace correspond to
runtime Celeritas classes and objects.

The following sections describe the members and their configuration options.
Note that most input classes (namespace ``inp``) match up with the runtime
classes that they help construct. Many of these definitions allow selection
between hard-coded C++ types via std::`variant`_ and optional types using
std::`optional`_.

.. _variant: https://en.cppreference.com/w/cpp/utility/variant
.. _optional: https://en.cppreference.com/w/cpp/utility/optional

Problems are loaded into the framework or application front end via :ref:`api_problem_setup`.

.. toctree::
   :maxdepth: 2

   input/framework.rst
   input/standalone.rst
   input/events.rst
   input/import.rst
   input/system.rst
   input/problem.rst
   input/model.rst
   input/physics.rst
   input/field.rst
   input/scoring.rst
   input/tracking.rst
   input/control.rst
   input/diagnostics.rst
