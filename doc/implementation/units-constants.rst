.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _units_constants:

Units and constants
===================

The default unit system in Celeritas is CGS with centimeter (cm), gram (g), second (s),
gauss (G), and kelvin (K) all having a value of unity. With these definitions,
densities can be defined in natural units of :math:`\mathrm{g}/\mathrm{cm}^3`,
and macroscopic cross sections are in units of :math:`\mathrm{cm}^{-1}`. See
the :ref:`units documentation <api_units>` for more descriptions of the core
unit system and the exactly defined values for SI units such as tesla.

Celeritas defines :ref:`constants <api_constants>` from a few different sources.
Mathematical constants are defined as truncated floating point values. Some
physical constants such as the speed of light, Planck's constant, and the
electron charge, have exact numerical values as specified by the SI unit system
:cite:`si-2019`. Other physical constants such as the atomic mass unit and electron
radius are derived from experimental measurements in CODATA 2018. Because the
reported constants are derived from regression fits to experimental data
points, some exactly defined physical relationships (such as the fine structure
:math:`\alpha = \frac{e^2}{2 \epsilon_0 h c}` are only approximate.

Unlike Geant4 and the CLHEP unit systems, Celeritas avoids using "natural"
units in its definitions. Although a natural unit system makes some
expressions easier to evaluate, it can lead to errors in the definition of
derivative constants and is fundamentally in conflict with consistent unit
systems such as SI. To enable special unit systems in harmony with the
native Celeritas unit system, the :ref:`Quantity <api_quantity>` class
stores quantities in another unit system with a compile-time constant that
allows their conversion back to native units. This allows, for example,
particles to represent their energy as MeV and charge as fractions of e but
work seamlessly with a field definition in native (macro-scale quantity) units.


.. _api_quantity:

Quantity
--------

Celeritas supports multiple simultaneous unit systems (e.g., atomic
scale/natural units working with a macro-scale but consistent unit system)
using the Quantity class and helper functions.

.. doxygenclass:: celeritas::Quantity
.. doxygenfunction:: celeritas::native_value_to
.. doxygenfunction:: celeritas::native_value_from(Quantity<UnitT, ValueT> quant) noexcept
.. doxygenfunction:: celeritas::value_as
.. doxygenfunction:: celeritas::zero_quantity
.. doxygenfunction:: celeritas::max_quantity
.. doxygenfunction:: celeritas::neg_max_quantity

An example quantity uses :math:`2\pi` as the unit type to allow integral values
for turns. Using a Quantity also allows us to override the ``sincos`` function
to use ``sincospi`` under the hood for improved precision.

.. doxygentypedef:: celeritas::Turn
.. doxygenfunction:: celeritas::sincos(Turn r, real_type* sinv, real_type* cosv)

.. _api_units:

Units
-----

.. doxygennamespace:: celeritas::units
   :no-link:

.. _api_constants:

Constants
---------

.. doxygennamespace:: celeritas::constants
   :no-link:
