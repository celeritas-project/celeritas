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
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert to Celeritas volume ID from geant4 transportation world.
 */
GeantVolumeMapper::GeantVolumeMapper(GeantGeoParams const& geo) : geo_{geo} {}

//---------------------------------------------------------------------------//
/*!
 * Find the Celeritas (\c VolumeParams ) volume ID for a Geant4 volume.
 *
 * This will warn if the name's extension had to be changed to match the
 * volume; and it will return an empty ID if no match was found.
 */
VolumeId GeantVolumeMapper::operator()(G4LogicalVolume const& lv)
{
    // TODO: move pointer->id mapping to GeantGeoParams
    VolumeId id = geo_.from_impl_volume(geo_.find_volume(&lv));
    if (id)
    {
        // Volume is mapped from an externally loaded Geant4 geometry
        return id;
    }

    // Get geant volume ID and corresponding label
    id = geo_.from_impl_volume(geo_.geant_to_id(lv));
    CELER_VALIDATE(id,
                   << "logical volume '" << lv.GetName()
                   << "' is not in the tracking volume");
    // TODO: volume ID should correspond one-to-one?
    auto const& label = geo_.impl_volumes().at(geo_.to_impl_volume(id));

    // Compare Geant4 label to main geometry label
    auto const& volumes = geo_.impl_volumes();
    if (auto id = geo_.from_impl_volume(volumes.find_exact(label)))
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
        return geo_.from_impl_volume(all_ids.front());
    }
    else if (all_ids.size() > 1)
    {
        CELER_LOG(warning)
            << "Multiple volumes '"
            << join(all_ids.begin(),
                    all_ids.end(),
                    "', '",
                    [&volumes](ImplVolumeId v) { return volumes.at(v); })
            << "' match the Geant4 volume '" << label
            << "' without extension: returning the last one";
        return geo_.from_impl_volume(all_ids.back());
    }

    // No match
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
