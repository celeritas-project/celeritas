//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantVolumeMapper.cc
//---------------------------------------------------------------------------//
#include "GeantVolumeMapper.hh"

#include <G4LogicalVolume.hh>

#include "celeritas_cmake_strings.h"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find the celeritas (VecGeom/ORANGE) volume ID for a Geant4 volume.
 *
 * This will warn if the name's extension had to be changed to match the
 * volume; and it will return an empty ID if no match was found.
 */
VolumeId GeantVolumeMapper::operator()(G4LogicalVolume const& lv) const
{
    if (VolumeId id = geo.find_volume(&lv))
    {
        // Volume is mapped from an externally loaded Geant4 geometry
        return id;
    }

    // Convert volume name to GPU geometry ID
    auto label = Label::from_geant(lv.GetName());
    if (label.ext.empty())
    {
        // Label doesn't have a pointer address attached: we probably need
        // to regenerate to match the exported GDML file
        label = Label::from_geant(make_gdml_name(lv));
    }

    if (auto id = geo.find_volume(label))
    {
        // Exact match
        return id;
    }

    // Fall back to skipping the extension: look for all possible matches
    auto all_ids = geo.find_volumes(label.name);
    if (all_ids.size() == 1)
    {
        CELER_LOG(warning) << "Failed to exactly match " << celeritas_core_geo
                           << " volume from Geant4 volume '" << lv.GetName()
                           << "'@" << static_cast<void const*>(&lv)
                           << "; found '" << geo.id_to_label(all_ids.front())
                           << "' by omitting the extension";
        return all_ids.front();
    }

    // Try regenerating the name even if we *did* have a pointer
    // address attached (in case an original GDML volume name already
    // had a pointer suffix and LoadGdml added another)
    label = Label::from_geant(make_gdml_name(lv));
    all_ids = geo.find_volumes(label.name);
    if (all_ids.size() > 1)
    {
        CELER_LOG(warning)
            << "Multiple volumes '"
            << join(all_ids.begin(),
                    all_ids.end(),
                    "', '",
                    [&geo = this->geo](VolumeId v) {
                        return geo.id_to_label(v);
                    })
            << "' match the Geant4 volume with extension omitted: returning "
               "the last one";
        return all_ids.back();
    }
    else if (all_ids.empty())
    {
        return {};
    }
    return all_ids.front();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
