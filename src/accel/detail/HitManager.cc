//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.cc
//---------------------------------------------------------------------------//
#include "HitManager.hh"

#include <utility>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Label.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/detail/GeantVolumeVisitor.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/io/ImportVolume.hh"
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
    , locate_touchable_(setup.locate_touchable)
{
    CELER_EXPECT(setup.enabled);

    // Convert setup options to step data
    selection_.energy_deposition = setup.energy_deposition;
    update_selection(&selection_.points[StepPoint::pre], setup.pre);
    update_selection(&selection_.points[StepPoint::post], setup.post);
    if (locate_touchable_)
    {
        selection_.points[StepPoint::pre].pos = true;
        selection_.points[StepPoint::pre].dir = true;
    }

    // Logical volumes to pass to hit processor
    std::vector<G4LogicalVolume*> geant_vols;

    // Helper class to extract GDML names+labels from Geant4 volume
    GeantVolumeVisitor visitor(true);

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
        if (label.ext.empty())
        {
            // Label doesn't have a pointer address attached: we probably need
            // to regenerate to match the exported GDML file
            label = Label::from_geant(visitor.generate_name(*lv));
        }

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
            else
            {
                // Try regenerating the name even if we *did* have a pointer
                // address attached (in case the original volume name already
                // had a pointer suffix and we added another)
                label = Label::from_geant(visitor.generate_name(*lv));
                id = geo.find_volume(label.name);
            }
        }
        CELER_VALIDATE(id,
                       << "failed to find " << celeritas_geometry
                       << " volume corresponding to Geant4 volume '"
                       << lv->GetName() << "'");

        // Add Geant4 volume and corresponding volume ID to list
        geant_vols.push_back(lv);
        vecgeom_vols_.push_back(id);
    }
    CELER_VALIDATE(!vecgeom_vols_.empty(),
                   << "no sensitive detectors were found");

    // Hit processors *must* be allocated on the thread they're used because of
    // geant4 thread-local SDs. There must be one per thread.
    auto* run_man = G4RunManager::GetRunManager();
    CELER_VALIDATE(run_man,
                   << "G4RunManager was not created before setting up "
                      "HitManager");
    processors_.resize(celeritas::get_num_threads(*run_man));

    geant_vols_ = std::make_shared<std::vector<G4LogicalVolume*>>(
        std::move(geant_vols));
    CELER_ENSURE(geant_vols_ && geant_vols_->size() == vecgeom_vols_.size());
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
    auto&& process_hits = this->get_local_hit_processor();
    process_hits(data);
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void HitManager::execute(StateDeviceRef const& data)
{
    auto&& process_hits = this->get_local_hit_processor();
    process_hits(data);
}

//---------------------------------------------------------------------------//
/*!
 * Destroy local data to avoid Geant4 crashes.
 *
 * This deallocates the local hit processor on the Geant4 thread that was
 * active when allocating it. This is necessary because Geant4 has thread-local
 * allocators that crash if trying to deallocate data allocated on another
 * thread.
 */
void HitManager::finalize()
{
    CELER_LOG_LOCAL(debug) << "Deallocating hit processor";
    int local_thread = G4Threading::G4GetThreadId();
    CELER_ASSERT(static_cast<std::size_t>(local_thread) < processors_.size());
    processors_[local_thread].reset();
}

//---------------------------------------------------------------------------//
/*!
 * Ensure the local hit processor exists, and return it.
 */
HitProcessor& HitManager::get_local_hit_processor()
{
    int local_thread = G4Threading::G4GetThreadId();
    CELER_ASSERT(static_cast<std::size_t>(local_thread) < processors_.size());
    if (CELER_UNLIKELY(!processors_[local_thread]))
    {
        CELER_LOG_LOCAL(debug) << "Allocating hit processor";
        // Allocate the hit processor locally
        processors_[local_thread] = std::make_unique<HitProcessor>(
            geant_vols_, selection_, locate_touchable_);
    }
    return *processors_[local_thread];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
