//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.cc
//---------------------------------------------------------------------------//
#include "HitManager.hh"

#include <utility>
#include <G4GDMLWriteStructure.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Label.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "accel/SetupOptions.hh"

#include "HitProcessor.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
void update_selection(StepPointSelection* selection,
                      SDSetupOptions::StepPoint const& options)
{
    selection->time = options.global_time;
    selection->pos = options.position;
    selection->energy = options.kinetic_energy;
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Map detector IDs on construction.
 */
HitManager::HitManager(GeoParams const& geo, SDSetupOptions const& setup)
    : nonzero_energy_deposition_(setup.ignore_zero_deposition)
{
    CELER_EXPECT(setup.enabled);

    // Convert setup options to step data
    selection_.energy_deposition = setup.energy_deposition;
    update_selection(&selection_.points[StepPoint::pre], setup.pre);
    update_selection(&selection_.points[StepPoint::post], setup.post);
    if (setup.locate_touchable)
    {
        selection_.points[StepPoint::pre].pos = true;
    }

    // Logical volumes to pass to hit processor
    std::vector<G4LogicalVolume*> lv_with_sd;

    // Helper class to extract GDML names+labels from Geant4 volume
    G4GDMLWriteStructure temp_writer;

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
        auto label
            = Label::from_geant(temp_writer.GenerateName(lv->GetName(), lv));
        auto id = geo.find_volume(label);
        if (!id)
        {
            // Fallback to skipping the extension
            id = geo.find_volume(label.name);
            if (id)
            {
                CELER_LOG(warning)
                    << "Failed to find " << celeritas_geometry
                    << " volume corresponding to Geant4 volume '"
                    << lv->GetName() << "'; found '" << geo.id_to_label(id)
                    << "' by omitting the extension";
            }
        }
        CELER_VALIDATE(id,
                       << "failed to find " << celeritas_geometry
                       << " volume corresponding to Geant4 volume '"
                       << lv->GetName() << "'");

        // Add Geant4 volume and corresponding volume ID to list
        lv_with_sd.push_back(lv);
        vecgeom_vols_.push_back(id);
    }
    CELER_VALIDATE(!vecgeom_vols_.empty(),
                   << "no sensitive detectors were found");

    process_hits_ = std::make_unique<HitProcessor>(
        std::move(lv_with_sd), selection_, setup.locate_touchable);
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

    result.nonzero_energy_deposition = nonzero_energy_deposition_;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void HitManager::execute(StateHostRef const& data)
{
    copy_steps(&steps_, data);
    if (steps_)
    {
        (*process_hits_)(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void HitManager::execute(StateDeviceRef const& data)
{
    copy_steps(&steps_, data);
    if (steps_)
    {
        (*process_hits_)(steps_);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
