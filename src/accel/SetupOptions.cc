//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include "celeritas/ext/GeantGeoUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find volumes by name for SDSetupOptions.
 */
std::unordered_set<G4LogicalVolume const*>
FindVolumes(std::unordered_set<std::string> names)
{
    return find_geant_volumes(std::move(names));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
