//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Units.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Types.hh"

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
 * be precisely defined. (E.g. you cannot define both MeV as 1 and Joule as 1.)
 * To express quantities in another system of units, e.g. MeV, use the Quantity
 * class.
 *
 * See also:
 *  - \c Constants.hh for constants defined in this unit system
 *  - \c physics/base/Units.hh for unit systems used by the physics
 *
 * Additionally:
 * - radians are used for measures of angle (unitless)
 * - steradians are used for measures of solid angle (unitless)
 */

//!@{
//! Units with numerical value defined to be 1
constexpr real_type centimeter = 1.; // Length
constexpr real_type gram       = 1.; // Mass
constexpr real_type second     = 1.; // Time
constexpr real_type coulomb    = 1.; // Charge
constexpr real_type kelvin     = 1.; // Temperature
//!@}

//!@{
//! Exact unit transformations for SI units
constexpr real_type meter    = 100 * centimeter;
constexpr real_type kilogram = 1000 * gram;
constexpr real_type newton   = kilogram * meter / (second * second);
constexpr real_type joule    = newton * meter;
constexpr real_type ampere   = coulomb / second;
constexpr real_type volt     = joule / coulomb;
constexpr real_type tesla    = kilogram / (coulomb * second);
constexpr real_type farad    = coulomb / volt;
//!@}

//!@{
//! Other units
constexpr real_type millimeter = 0.1 * centimeter;
constexpr real_type barn       = 1e-24 * centimeter * centimeter;
//!@}

//---------------------------------------------------------------------------//
} // namespace units
} // namespace celeritas
