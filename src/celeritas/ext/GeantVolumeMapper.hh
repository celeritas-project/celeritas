//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantVolumeMapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Config.hh"

#include "corecel/io/Label.hh"
#include "geocel/GeoParamsInterface.hh"
#include "celeritas/Types.hh"

// Geant4 forward declaration
class G4LogicalVolume;  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map a Geant4 logical volume to a Celeritas volume ID.
 */
class GeantVolumeMapper
{
  public:
    // Convert to target geometry from geant4 transportation world
    explicit GeantVolumeMapper(GeoParamsInterface const& tgt);

    // Convert a volume; null if not found; warn if inexact match
    VolumeId operator()(G4LogicalVolume const&);

  private:
    GeoParamsInterface const& geo_;
    std::vector<Label> labels_;
};

#if !CELERITAS_USE_GEANT4
inline GeantVolumeMapper::GeantVolumeMapper(GeoParamsInterface const& geo)
    : geo_{geo}
{
    CELER_DISCARD(labels_);
    CELER_DISCARD(geo_);
    CELER_NOT_CONFIGURED("Geant4");
}

inline VolumeId GeantVolumeMapper::operator()(G4LogicalVolume const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
