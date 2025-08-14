//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/SensDetInserter.cc
//---------------------------------------------------------------------------//
#include "SensDetInserter.hh"

#include <G4VSensitiveDetector.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with references to the inserted data.
 */
SensDetInserter::SensDetInserter(SetLV const& skip_volumes,
                                 MapIdLv* found,
                                 VecLV* missing)
    : skip_volumes_{skip_volumes}, found_{found}, missing_{missing}
{
    CELER_EXPECT(found_);
    CELER_EXPECT(missing_);
}

//---------------------------------------------------------------------------//
/*!
 * Save a sensitive detector.
 */
void SensDetInserter::operator()(G4LogicalVolume const* lv,
                                 G4VSensitiveDetector const* sd)
{
    CELER_EXPECT(lv);
    CELER_EXPECT(sd);

    if (VolumeId id = this->insert_impl(lv))
    {
        CELER_LOG(debug) << "Mapped sensitive detector \"" << sd->GetName()
                         << "\" on logical volume " << PrintableLV{lv}
                         << " to volume ID " << id.get();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Save a sensitive detector.
 */
void SensDetInserter::operator()(G4LogicalVolume const* lv)
{
    CELER_EXPECT(lv);

    if (VolumeId id = this->insert_impl(lv))
    {
        CELER_LOG(debug) << "Mapped unspecified detector on logical volume "
                         << PrintableLV{lv} << " to volume ID " << id.get();
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

    VolumeId id = to_vol_id_(lv);
    if (!id)
    {
        missing_->push_back(lv);
        return {};
    }

    // Add Geant4 volume and corresponding volume ID to list
    auto [iter, inserted] = found_->insert({id, lv});

    if (CELER_UNLIKELY(!inserted))
    {
        if (iter->second != lv)
        {
            // This shouldn't be possible now
            CELER_LOG(error)
                << "Canonical volume " << id.get()
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
