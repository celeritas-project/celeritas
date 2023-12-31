//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Units.hh
//! \brief Unit definitions
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Types.hh"

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
/*!
 * \namespace units
 * Units and constants in Celeritas for macro-scale quantities.
 *
 * The following units have numerical values of 1 in this system:
 * - cm for standard unit of length
 * - s for standard unit of time
 * - g for standard unit of mass.
 *
 * Unless otherwise specified, this unit system is used for input and
 * output numerical values. They are meant for macro-scale quantities coupling
 * the different code components of Celeritas.
 *
 * \note This system of units should be fully consistent so that constants can
 * be precisely defined. (E.g., you cannot define both MeV as 1 and Joule
 * as 1.) To express quantities in another system of units, such as MeV and
 * "natural" units, use the Quantity class.
 *
 * See also:
 *  - \c Constants.hh for constants defined in this unit system
 *  - \c physics/base/Units.hh for unit systems used by the physics
 *
 * Additionally:
 * - radians are used for measures of angle (unitless)
 * - steradians are used for measures of solid angle (unitless)
 */

#define CELER_ICRT inline constexpr real_type

#if CELERITAS_UNITS == CELERITAS_UNITS_CGS
//!@{
//! \name Units with numerical value defined to be 1
CELER_ICRT centimeter = 1;  //!< Length
CELER_ICRT gram = 1;  //!< Mass
CELER_ICRT second = 1;  //!< Time
CELER_ICRT gauss = 1;  //!< Field strength
CELER_ICRT kelvin = 1;  //!< Temperature
//!@}

//!@{
//! \name Exact unit transformations to SI units
CELER_ICRT meter = 100 * centimeter;
CELER_ICRT kilogram = 1000 * gram;
CELER_ICRT tesla = 10000 * gauss;
//!@}

//!@{
//! \name Exact unit transformations using SI unit definitions
CELER_ICRT newton = kilogram * meter / (second * second);
CELER_ICRT joule = newton * meter;
CELER_ICRT coulomb = kilogram / (tesla * second);
CELER_ICRT ampere = coulomb / second;
CELER_ICRT volt = joule / coulomb;
CELER_ICRT farad = coulomb / volt;
//!@}

//!@{
//! \name CLHEP units
CELER_ICRT millimeter = real_type(0.1) * centimeter;
CELER_ICRT nanosecond = real_type(1e-9) * second;
//!@}

#elif CELERITAS_UNITS == CELERITAS_UNITS_CLHEP

CELER_ICRT millimeter = 1;  //!< Length
CELER_ICRT megaelectronvolt = 1;  //!< Mass
CELER_ICRT nanosecond = 1;  //!< Time
CELER_ICRT e_electron = 1;  //!< Charge
CELER_ICRT kelvin = 1;  //!< Temperature

CELER_ICRT coulomb = e_electron / 1.602176634e-19;  //! Value from SI 2019
CELER_ICRT volt = real_type(1e-6) * (megaelectronvolt / e_electron);
CELER_ICRT joule = coulomb * volt;

CELER_ICRT second = real_type(1e9) * nanosecond;
CELER_ICRT meter = 1000 * millimeter;

CELER_ICRT ampere = coulomb / second;
CELER_ICRT farad = coulomb / volt;
CELER_ICRT kilogram = joule * (second / meter) * (second / meter);
CELER_ICRT tesla = volt * second / (meter * meter);
CELER_ICRT newton = joule / meter;

//!@{
//! \name CGS-specific units
CELER_ICRT centimeter = real_type(10) * millimeter;
CELER_ICRT gram = real_type(1e-3) * kilogram;
CELER_ICRT gauss = real_type(1e-4) * tesla;
//!@}

#endif

//!@{
//! \name Other common units
CELER_ICRT femtometer = real_type(1e-13) * centimeter;
CELER_ICRT barn = real_type(1e-24) * centimeter * centimeter;
//!@}

#undef CELER_ICRT

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
