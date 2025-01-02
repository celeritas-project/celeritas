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

    auto const& volumes = geo.volumes();
    if (auto id = volumes.find_exact(label))
    {
        // Exact match
        return id;
    }

    // Fall back to skipping the extension: look for all possible matches
    auto all_ids = volumes.find_all(label.name);
    if (all_ids.size() == 1)
    {
        CELER_LOG(warning) << "Failed to exactly match " << celeritas_core_geo
                           << " volume from Geant4 volume '" << lv.GetName()
                           << "'@" << static_cast<void const*>(&lv)
                           << "; found '" << volumes.at(all_ids.front())
                           << "' by omitting the extension";
        return all_ids.front();
    }

    // Try regenerating the name even if we *did* have a pointer
    // address attached (in case an original GDML volume name already
    // had a pointer suffix and LoadGdml added another)
    label = Label::from_geant(make_gdml_name(lv));
    all_ids = volumes.find_all(label.name);
    if (all_ids.size() > 1)
    {
        CELER_LOG(warning)
            << "Multiple volumes '"
            << join(all_ids.begin(),
                    all_ids.end(),
                    "', '",
                    [&volumes](VolumeId v) { return volumes.at(v); })
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
