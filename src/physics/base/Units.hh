//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Units.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
/*!
 * \namespace units
 * Description of the system of units in Celeritas.
 *
 * Celeritas uses a CGS-based system with:
 * - Natural units (c = 1)
 * - MeV as the standard unit for energy
 */

//@{
//! Length
constexpr real_type centimeter = 1.;
constexpr real_type millimeter = centimeter / 10.;
constexpr real_type meter      = 100. * centimeter;
//@}

//@{
//! Mass
constexpr real_type gram      = 1.;
constexpr real_type milligram = 1e-3 * gram;
constexpr real_type kilogram  = 1e3 * gram;
//@}

//@{
//! Time
constexpr real_type second      = 1.;
constexpr real_type millisecond = 1.e-3 * second;
constexpr real_type microsecond = 1.e-6 * second;
constexpr real_type nanosecond  = 1.e-9 * second;
//@}

//@{
//! Energy
constexpr real_type mega_electron_volt = 1.;
constexpr real_type electron_volt      = 1.e-6 * mega_electron_volt;
constexpr real_type kilo_electron_volt = 1.e-3 * mega_electron_volt;
constexpr real_type giga_electron_volt = 1.e+3 * mega_electron_volt;
constexpr real_type tera_electron_volt = 1.e+6 * mega_electron_volt;
constexpr real_type peta_electron_volt = 1.e+9 * mega_electron_volt;
//@}

//@{
//! Area
constexpr real_type barn      = 1.e-24 * centimeter * centimeter;
constexpr real_type millibarn = 1.e-3 * barn;
//@}

//@{
//! Derived quantities
constexpr real_type speed_of_light    = 1.; // Natural unit
constexpr real_type speed_of_light_sq = 1.;
//@}

//---------------------------------------------------------------------------//
} // namespace units
} // namespace celeritas

//---------------------------------------------------------------------------//
