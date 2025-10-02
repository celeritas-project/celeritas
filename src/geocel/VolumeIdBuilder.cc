//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeIdBuilder.cc
//---------------------------------------------------------------------------//
#include "VolumeIdBuilder.hh"

#include "corecel/io/Join.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/Logger.hh"

#include "GeantGeoParams.hh"
#include "GeantGeoUtils.hh"
#include "VolumeParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from global geometry.
 */
VolumeIdBuilder::VolumeIdBuilder()
    : VolumeIdBuilder{global_volumes().lock().get(),
                      global_geant_geo().lock().get()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Map from a string using VolumeParams.
 */
VolumeId VolumeIdBuilder::operator()(std::string const& s) const
{
    return (*this)(Label{s, {}});
}

//---------------------------------------------------------------------------//
/*!
 * Map from a label using VolumeParams.
 */
VolumeId VolumeIdBuilder::operator()(Label const& label) const
{
    CELER_EXPECT(volumes_);

    auto const& vol_labels = volumes_->volume_labels();
    if (auto id = vol_labels.find_exact(label))
    {
        // Exact match
        return id;
    }

    // Fall back to skipping the extension: look for all possible matches
    auto all_ids = vol_labels.find_all(label.name);
    if (all_ids.size() == 1)
    {
        if (!label.ext.empty())
        {
            CELER_LOG(warning)
                << "Failed to exactly match canonical volume from volume '"
                << label << "'; found '" << vol_labels.at(all_ids.front())
                << "' by ignoring extensions";
        }
        return all_ids.front();
    }
    if (all_ids.size() > 1)
    {
        CELER_LOG(warning)
            << "Multiple volumes '"
            << join(all_ids.begin(),
                    all_ids.end(),
                    "', '",
                    [&vol_labels](VolumeId v) { return vol_labels.at(v); })
            << "' match the name '" << label.name
            << "': returning the last one";
        return all_ids.back();
    }

    CELER_LOG(error) << "Failed to find volume corresponding to label '"
                     << label << "'";
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Map from Geant4 volume pointer using geant_geo.
 */
VolumeId VolumeIdBuilder::operator()(G4LogicalVolume const* lv) const
{
    CELER_EXPECT(geant_geo_);
    if (!lv)
    {
        CELER_LOG(warning) << "Invalid logical volume: <null>";
        return {};
    }

    if (auto result = geant_geo_->geant_to_id(*lv))
    {
        CELER_ENSURE(!volumes_ || result < volumes_->num_volumes());
        return result;
    }

    CELER_LOG(error) << "logical volume " << StreamableLV{lv}
                     << " is not in the tracking geometry";
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
