//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include "geocel/GeantGeoUtils.hh"

#include "ExceptionConverter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find volumes by name for SDSetupOptions.
 *
 * Example:
 * \code
   setup.sd.force_volumes = FindVolumes({"foo", "bar"});
 * \endcode
 */
std::unordered_set<G4LogicalVolume const*>
FindVolumes(std::unordered_set<std::string> names)
{
    ExceptionConverter call_g4exception{"celer0006"};
    std::unordered_set<G4LogicalVolume const*> result;
    CELER_TRY_HANDLE(result = find_geant_volumes(std::move(names)),
                     call_g4exception);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
