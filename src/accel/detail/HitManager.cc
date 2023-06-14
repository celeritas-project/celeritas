//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.cc
//---------------------------------------------------------------------------//
#include "HitManager.hh"

#include <map>
#include <utility>
#include <G4LogicalVolumeStore.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4VSensitiveDetector.hh>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantGeoUtils.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/GeantVolumeMapper.hh"
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
HitManager::HitManager(GeoParams const& geo,
                       SDSetupOptions const& setup,
                       StreamId::size_type num_streams)
    : nonzero_energy_deposition_(setup.ignore_zero_deposition)
    , locate_touchable_(setup.locate_touchable)
{
    CELER_EXPECT(setup.enabled);
    CELER_EXPECT(num_streams > 0);

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
    std::map<G4LogicalVolume const*, VolumeId> found_lv;
    // LVs that we can't correlate to Celeritas volume labels
    VecLV missing_lv;

    // Volume mapper and helper lambda for adding SDs
    GeantVolumeMapper g4_to_celer{geo};
    auto add_volume
        = [&](G4LogicalVolume const* lv, G4VSensitiveDetector const* sd) {
              if (setup.skip_volumes.count(lv))
              {
                  CELER_LOG(debug)
                      << "Skipping automatic SD callback for logical volume '"
                      << lv->GetName() << "' due to user option";
                  return;
              }

              auto id = lv ? g4_to_celer(*lv) : VolumeId{};
              if (!id)
              {
                  missing_lv.push_back(lv);
                  return;
              }
              auto msg = CELER_LOG(debug);
              msg << "Mapped ";
              if (sd)
              {
                  msg << "sensitive detector '" << sd->GetName() << '\'';
              }
              else
              {
                  msg << "unknown sensitive detector";
              }
              msg << " on logical volume '" << PrintableLV{lv} << "' to "
                  << celeritas_core_geo << " volume '" << geo.id_to_label(id)
                  << "' (ID=" << id.unchecked_get() << ')';

              // Add Geant4 volume and corresponding volume ID to list
              found_lv.insert({lv, id});
          };

    // Loop over all logical volumes and map detectors to Volume IDs
    for (G4LogicalVolume const* lv : *G4LogicalVolumeStore::GetInstance())
    {
        CELER_ASSERT(lv);

        // Check for sensitive detectors attached to the master thread
        G4VSensitiveDetector* sd = lv->GetSensitiveDetector();
        if (sd)
        {
            add_volume(lv, sd);
        }
    }

    // Loop over user-specified G4LV
    for (G4LogicalVolume const* lv : setup.force_volumes)
    {
        add_volume(lv, nullptr);
    }

    CELER_VALIDATE(
        missing_lv.empty(),
        << "failed to find unique " << celeritas_core_geo
        << " volume(s) corresponding to Geant4 volume(s) "
        << join_stream(missing_lv.begin(),
                       missing_lv.end(),
                       ", ",
                       [](std::ostream& os, G4LogicalVolume const* lv) {
                           os << '\'' << PrintableLV{lv} << '\'';
                       }));
    CELER_VALIDATE(!found_lv.empty(), << "no sensitive detectors were found");

    // Hit processors *must* be allocated on the thread they're used because of
    // geant4 thread-local SDs. There must be one per thread.
    processors_.resize(num_streams);

    // Unfold map into LV/ID vectors
    VecLV geant_vols;
    geant_vols.reserve(found_lv.size());
    vecgeom_vols_.reserve(found_lv.size());
    for (auto&& [lv, id] : found_lv)
    {
        geant_vols.push_back(lv);
        vecgeom_vols_.push_back(id);
    }
    geant_vols_ = std::make_shared<VecLV>(std::move(geant_vols));

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
void HitManager::process_steps(HostStepState state)
{
    auto&& process_hits = this->get_local_hit_processor(state.stream_id);
    process_hits(state.steps);
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void HitManager::process_steps(DeviceStepState state)
{
    auto&& process_hits = this->get_local_hit_processor(state.stream_id);
    process_hits(state.steps);
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
void HitManager::finalize(StreamId sid)
{
    CELER_EXPECT(sid < processors_.size());
    CELER_LOG_LOCAL(debug) << "Deallocating hit processor (stream "
                           << sid.get() << ")";
    processors_[sid.unchecked_get()].reset();
}

//---------------------------------------------------------------------------//
/*!
 * Ensure the local hit processor exists, and return it.
 */
HitProcessor& HitManager::get_local_hit_processor(StreamId sid)
{
    CELER_EXPECT(sid < processors_.size());

    if (CELER_UNLIKELY(!processors_[sid.unchecked_get()]))
    {
        CELER_LOG_LOCAL(debug)
            << "Allocating hit processor (stream " << sid.get() << ")";
        // Allocate the hit processor locally
        processors_[sid.unchecked_get()] = std::make_unique<HitProcessor>(
            geant_vols_, selection_, locate_touchable_);
    }
    return *processors_[sid.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
