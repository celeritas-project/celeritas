//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Units.hh
//! \brief Unit definitions
//---------------------------------------------------------------------------//
#pragma once

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
 * as 1.) To express quantities in another system of units, e.g. MeV, use the
 * Quantity class.
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

//!@{
//! \name Units with numerical value defined to be 1
CELER_ICRT centimeter = 1;  //!< Length
CELER_ICRT gram = 1;  //!< Mass
CELER_ICRT second = 1;  //!< Time
CELER_ICRT gauss = 1;  //!< Field strength
CELER_ICRT kelvin = 1;  //!< Temperature
//!@}

//!@{
//! \name Exact unit transformations for SI units
CELER_ICRT meter = 100 * centimeter;
CELER_ICRT kilogram = 1000 * gram;
CELER_ICRT tesla = 10000 * gauss;
CELER_ICRT newton = kilogram * meter / (second * second);
CELER_ICRT joule = newton * meter;
CELER_ICRT coulomb = kilogram / (tesla * second);
CELER_ICRT ampere = coulomb / second;
CELER_ICRT volt = joule / coulomb;
CELER_ICRT farad = coulomb / volt;
//!@}

//!@{
//! \name Other common units
CELER_ICRT millimeter = real_type(0.1) * centimeter;
CELER_ICRT femtometer = real_type(1e-13) * centimeter;
CELER_ICRT barn = real_type(1e-24) * centimeter * centimeter;
//!@}

#undef CELER_ICRT

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
