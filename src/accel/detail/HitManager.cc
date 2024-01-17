//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.cc
//---------------------------------------------------------------------------//
#include "HitManager.hh"

#include <unordered_map>
#include <utility>
#include <G4LogicalVolumeStore.hh>
#include <G4ParticleTable.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "accel/SetupOptions.hh"

#include "HitProcessor.hh"
#include "SensDetInserter.hh"

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
    selection->dir = options.direction;
    selection->energy = options.kinetic_energy;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Map detector IDs on construction.
 */
HitManager::HitManager(GeoParams const& geo,
                       ParticleParams const& par,
                       SDSetupOptions const& setup,
                       StreamId::size_type num_streams)
    : nonzero_energy_deposition_(setup.ignore_zero_deposition)
    , locate_touchable_(setup.locate_touchable)
{
    CELER_EXPECT(setup.enabled);
    CELER_EXPECT(num_streams > 0);

    // Convert setup options to step data
    selection_.particle = setup.track;
    selection_.energy_deposition = setup.energy_deposition;
    update_selection(&selection_.points[StepPoint::pre], setup.pre);
    update_selection(&selection_.points[StepPoint::post], setup.post);
    if (locate_touchable_)
    {
        selection_.points[StepPoint::pre].pos = true;
        selection_.points[StepPoint::pre].dir = true;
    }

    // Hit processors *must* be allocated on the thread they're used because of
    // geant4 thread-local SDs. There must be one per thread.
    processors_.resize(num_streams);

    // Map detector volumes
    this->setup_volumes(geo, setup);

    if (setup.track)
    {
        this->setup_particles(par);
    }

    CELER_ENSURE(setup.track == !this->particles_.empty());
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
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
void HitManager::setup_volumes(GeoParams const& geo,
                               SDSetupOptions const& setup)
{
    // Helper for inserting volumes
    SensDetInserter::MapIdLv found_id_lv;
    SensDetInserter::VecLV missing_lv;
    SensDetInserter insert_volume(
        geo, setup.skip_volumes, &found_id_lv, &missing_lv);

    // Loop over all logical volumes and map detectors to Volume IDs
    for (G4LogicalVolume const* lv : *G4LogicalVolumeStore::GetInstance())
    {
        if (lv)
        {
            if (G4VSensitiveDetector* sd = lv->GetSensitiveDetector())
            {
                // Sensitive detector is attached to the master thread
                insert_volume(lv, sd);
            }
        }
    }

    // Loop over user-specified G4LV
    for (G4LogicalVolume const* lv : setup.force_volumes)
    {
        insert_volume(lv);
    }

    CELER_VALIDATE(
        missing_lv.empty(),
        << "failed to find unique " << celeritas_core_geo
        << " volume(s) corresponding to Geant4 volume(s) "
        << join_stream(missing_lv.begin(),
                       missing_lv.end(),
                       ", ",
                       [](std::ostream& os, G4LogicalVolume const* lv) {
                           os << '"' << lv->GetName() << '"';
                       })
        << " while mapping sensitive detectors");
    CELER_VALIDATE(!found_id_lv.empty(),
                   << "no sensitive detectors were found");

    // Unfold map into LV/ID vectors
    VecLV geant_vols;
    geant_vols.reserve(found_id_lv.size());
    vecgeom_vols_.reserve(found_id_lv.size());
    for (auto&& [id, lv] : found_id_lv)
    {
        geant_vols.push_back(lv);
        vecgeom_vols_.push_back(id);
    }
    geant_vols_ = std::make_shared<VecLV>(std::move(geant_vols));
}

//---------------------------------------------------------------------------//
void HitManager::setup_particles(ParticleParams const& par)
{
    CELER_EXPECT(selection_.particle);

    auto& g4particles = *G4ParticleTable::GetParticleTable();

    particles_.resize(par.size());
    std::vector<ParticleId> missing;
    for (auto pid : range(ParticleId{par.size()}))
    {
        int pdg = par.id_to_pdg(pid).get();
        if (G4ParticleDefinition* particle = g4particles.FindParticle(pdg))
        {
            particles_[pid.get()] = particle;
        }
        else
        {
            missing.push_back(pid);
        }
    }

    CELER_VALIDATE(missing.empty(),
                   << "failed to map Celeritas particles to Geant4: missing "
                   << join_stream(missing.begin(),
                                  missing.end(),
                                  ", ",
                                  [&par](std::ostream& os, ParticleId pid) {
                                      os << '"' << par.id_to_label(pid)
                                         << "\" (ID=" << pid.unchecked_get()
                                         << ", PDG="
                                         << par.id_to_pdg(pid).unchecked_get()
                                         << ")";
                                  }));
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
            geant_vols_, particles_, selection_, locate_touchable_);
    }
    return *processors_[sid.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
