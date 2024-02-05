//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantUnits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Quantities.hh"
#include "celeritas/UnitTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// CONSTANTS
//---------------------------------------------------------------------------//
//! Value of a unit Celeritas field in the CLHEP unit system
inline constexpr real_type clhep_field = 1
                                         / units::ClhepTraits::BField::value();
//! Value of a unit Celeritas time in the CLHEP unit system
inline constexpr real_type clhep_time = 1 / units::ClhepTraits::Time::value();

//---------------------------------------------------------------------------//
/*!
 * Convert Celeritas energy quantities to Geant4.
 *
 * The unit value should always be CLHEP::MeV which is defined to be unity.
 */
inline constexpr double
convert_to_geant(units::MevEnergy const& energy, double units)
{
    CELER_EXPECT(units == 1);
    return energy.value();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
