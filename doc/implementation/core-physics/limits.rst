.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _limits:

Step limits
===========

Currently there are only a few, hard-coded step limitation mechanics that apply
to a particle at each step. The ordering of these can affect the performance.

.. _limits_interaction:

Discrete interactions
---------------------

Most physics processes use pre-calculated cross sections that are tabulated and
interpolated.

.. doxygenclass:: celeritas::XsCalculator

Cross sections for each process are evaluated at the beginning of the step
along with range limiters.

.. doxygenfunction:: celeritas::calc_physics_step_limit

If undergoing an interaction, the process is sampled from the stored
beginning-of-step cross sections.

.. doxygenfunction:: celeritas::select_discrete_interaction

User-input
----------

See :cpp:class:`celeritas::inp::TrackingLimits`.

Range and energy loss
---------------------

See :ref:`em_slowing_down`.

Multiple scattering
-------------------

See :ref:`em_coulomb`.

.. doxygenclass:: celeritas::detail::UrbanMscSafetyStepLimit

.. doxygenclass:: celeritas::detail::UrbanMscMinimalStepLimit

When multiple scattering is in play, the "physical" step limit must be
converted to and from a "geometry" step limit before and after the propagation
(movement within geometry) step limiting calculation.

.. doxygenclass:: celeritas::detail::MscStepFromGeo
.. doxygenclass:: celeritas::detail::MscStepToGeo

Geometry
--------

See :ref:`api_propagation`.
