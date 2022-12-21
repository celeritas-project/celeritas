//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.cc
//---------------------------------------------------------------------------//
#include "HitManager.hh"

#include <G4LogicalVolumeStore.hh>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/Label.hh"
#include "celeritas/geo/GeoParams.hh"

#include "HitProcessor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Map detector IDs on construction.
 */
HitManager::HitManager(const GeoParams&     geo,
                       const StepSelection& selection,
                       const Options&       options)
    : selection_(selection), options_(options)
{
    selection_.event_id = true;

    // Logical volumes to pass to hit processor
    std::vector<G4LogicalVolume*> lv_with_sd;

    // Loop over all logical volumes
    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(lv_store);
    for (G4LogicalVolume* lv : *lv_store)
    {
        CELER_ASSERT(lv);

        // Check for sensitive detectors
        G4VSensitiveDetector* sd = lv->GetSensitiveDetector();
        if (!sd)
        {
            continue;
        }

        // Convert volume name to GPU geometry ID
        auto label = Label::from_geant(lv->GetName());
        auto id    = geo.find_volume(label);
        CELER_VALIDATE(id,
                       << "failed to find " << celeritas_geometry
                       << " volume corresponding to Geant4 volume " << label);

        // Add Geant4 volume and corresponding volume ID to list
        lv_with_sd.push_back(lv);
        vecgeom_vols_.push_back(id);
    }
    CELER_VALIDATE(!vecgeom_vols_.empty(),
                   << "no sensitive detectors were found");

    process_hits_ = std::make_unique<HitProcessor>(
        std::move(lv_with_sd), selection_, options_.locate_touchable);
}

//---------------------------------------------------------------------------//
//! Default destructor
HitManager::~HitManager() = default;

//---------------------------------------------------------------------------//
/*!
 * Map volume names to detector IDs and exclude tracks with no deposition.
 */
auto HitManager::filters() const -> Filters
{
    Filters result;

    for (auto didx : range<DetectorId::size_type>(vecgeom_vols_.size()))
    {
        result.detectors[vecgeom_vols_[didx]] = DetectorId{didx};
    }

    result.nonzero_energy_deposition = options_.nonzero_energy_deposition;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void HitManager::execute(StateHostRef const& data)
{
    copy_steps(&steps_, data);
    (*process_hits_)(steps_);
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void HitManager::execute(StateDeviceRef const& data)
{
    copy_steps(&steps_, data);
    (*process_hits_)(steps_);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
