//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantVolumeMapper.cc
//---------------------------------------------------------------------------//
#include "GeantVolumeMapper.hh"

#include <G4LogicalVolume.hh>

#include "corecel/Config.hh"

#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert to target geometry from geant4 transportation world.
 */
GeantVolumeMapper::GeantVolumeMapper(GeoParamsInterface const& geo)
    : world_{geant_world_volume()}, geo_{geo}
{
    CELER_VALIDATE(world_, << "world was not set up before mapping volumes");
}

//---------------------------------------------------------------------------//
/*!
 * Convert from Geant4 world *to* geometry interface ID.
 */
GeantVolumeMapper::GeantVolumeMapper(G4VPhysicalVolume const& world,
                                     GeoParamsInterface const& geo)
    : world_{&world}, geo_{geo}
{
}

//---------------------------------------------------------------------------//
/*!
 * Find the celeritas (VecGeom/ORANGE) volume ID for a Geant4 volume.
 *
 * This will warn if the name's extension had to be changed to match the
 * volume; and it will return an empty ID if no match was found.
 */
VolumeId GeantVolumeMapper::operator()(G4LogicalVolume const& lv)
{
    if (VolumeId id = geo_.find_volume(&lv))
    {
        // Volume is mapped from an externally loaded Geant4 geometry
        return id;
    }

    if (CELER_UNLIKELY(labels_.empty()))
    {
        // Lazily construct labels if lookup via pointer fails
        labels_ = make_logical_vol_labels(*world_);
    }

    // Convert volume name to GPU geometry ID
    CELER_VALIDATE(
        static_cast<std::size_t>(lv.GetInstanceID()) < labels_.size(),
        << "logical volume '" << lv.GetName()
        << "' is not in the world volume '" << world_->GetName() << "' ("
        << labels_.size() << " volumes found)");
    auto const& label = labels_[lv.GetInstanceID()];

    auto const& volumes = geo_.volumes();
    if (auto id = volumes.find_exact(label))
    {
        // Exact match
        return id;
    }

    // Fall back to skipping the extension: look for all possible matches
    auto all_ids = volumes.find_all(label.name);
    if (all_ids.size() == 1)
    {
        CELER_LOG(warning) << "Failed to exactly match " << cmake::core_geo
                           << " volume from Geant4 volume '" << label
                           << "'; found '" << volumes.at(all_ids.front())
                           << "' by omitting the extension";
        return all_ids.front();
    }
    else if (all_ids.size() > 1)
    {
        CELER_LOG(warning)
            << "Multiple volumes '"
            << join(all_ids.begin(),
                    all_ids.end(),
                    "', '",
                    [&volumes](VolumeId v) { return volumes.at(v); })
            << "' match the Geant4 volume '" << label
            << "' without extension: returning the last one";
        return all_ids.back();
    }

    // No match
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
