//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include <G4Threading.hh>

#include "celeritas/ext/GeantGeoUtils.hh"

#include "ExceptionConverter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find volumes by name for SDSetupOptions.
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
/*!
 * Get the thread ID.
 */
int GetThreadID()
{
    // Thread ID is -1 when running serially
    if (G4Threading::IsMultithreadedApplication())
    {
        return G4Threading::G4GetThreadId();
    }
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
