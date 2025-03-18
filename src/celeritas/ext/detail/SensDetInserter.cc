//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/SensDetInserter.cc
//---------------------------------------------------------------------------//
#include "SensDetInserter.hh"

#include <G4VSensitiveDetector.hh>

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Save a sensitive detector.
 */
void SensDetInserter::operator()(G4LogicalVolume const* lv,
                                 G4VSensitiveDetector const* sd)
{
    CELER_EXPECT(lv);
    CELER_EXPECT(sd);

    if (VolumeId id = insert_impl(lv))
    {
        CELER_LOG(debug) << "Mapped sensitive detector \"" << sd->GetName()
                         << "\" on logical volume " << PrintableLV{lv}
                         << " to " << cmake::core_geo << " volume \""
                         << geo_.volumes().at(id)
                         << "\" (ID=" << id.unchecked_get() << ')';
    }
}

//---------------------------------------------------------------------------//
/*!
 * Save a sensitive detector.
 */
void SensDetInserter::operator()(G4LogicalVolume const* lv)
{
    CELER_EXPECT(lv);

    if (VolumeId id = insert_impl(lv))
    {
        CELER_LOG(debug) << "Mapped unspecified detector on logical volume "
                         << PrintableLV{lv} << " to " << cmake::core_geo
                         << " volume \"" << geo_.volumes().at(id)
                         << "\" (ID=" << id.unchecked_get() << ')';
    }
}

//---------------------------------------------------------------------------//
VolumeId SensDetInserter::insert_impl(G4LogicalVolume const* lv)
{
    if (skip_volumes_.count(lv))
    {
        CELER_LOG(debug)
            << "Skipping automatic SD callback for logical volume \""
            << PrintableLV{lv} << "\" due to user option";
        return {};
    }

    auto id = lv ? g4_to_celer_(*lv) : VolumeId{};
    if (!id)
    {
        CELER_LOG(error) << "Failed to find " << cmake::core_geo
                         << " volume corresponding to Geant4 volume "
                         << PrintableLV{lv};
        missing_->push_back(lv);
        return {};
    }

    // Add Geant4 volume and corresponding volume ID to list
    auto [iter, inserted] = found_->insert({id, lv});

    if (CELER_UNLIKELY(!inserted))
    {
        if (iter->second != lv)
        {
            CELER_LOG(warning)
                << "Celeritas volume \"" << geo_.volumes().at(id)
                << "\" is mapped to two different volumes with "
                   "sensitive detectors: "
                << PrintableLV{lv} << " and " << PrintableLV{iter->second};
        }
        else
        {
            CELER_LOG(debug)
                << "Ignored duplicate logical volume " << PrintableLV{lv};
        }
    }

    return inserted ? id : VolumeId{};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
