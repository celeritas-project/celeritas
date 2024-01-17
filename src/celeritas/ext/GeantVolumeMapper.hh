//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantVolumeMapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "orange/GeoParamsInterface.hh"
#include "celeritas/Types.hh"

// Geant4 forward declaration
class G4LogicalVolume;  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map a Geant4 logical volume to a Celeritas volume ID.
 */
struct GeantVolumeMapper
{
    GeoParamsInterface const& geo;

    // Convert a volume; null if not found; warn if inexact match
    VolumeId operator()(G4LogicalVolume const&) const;
};

#if !CELERITAS_USE_GEANT4
inline VolumeId GeantVolumeMapper::operator()(G4LogicalVolume const&) const
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
