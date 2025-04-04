//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSd.cc
//---------------------------------------------------------------------------//
#include "GeantSd.hh"

#include <mutex>
#include <unordered_map>
#include <utility>
#include <G4LogicalVolumeStore.hh>
#include <G4ParticleTable.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "corecel/Config.hh"

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/inp/Scoring.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "GeantSetup.hh"

#include "detail/HitProcessor.hh"
#include "detail/SensDetInserter.hh"

using celeritas::detail::SensDetInserter;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
void update_selection(StepPointSelection* selection,
                      inp::GeantSdStepPointAttributes const& options)
{
    selection->time = options.global_time;
    selection->pos = options.position;
    selection->dir = options.direction;
    selection->energy = options.kinetic_energy;
    if (options.touchable)
    {
        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            CELER_LOG(warning) << "Using less accurate navigator-based "
                                  "reconstruction for ORANGE geometry";
            selection->pos = true;
            selection->dir = true;
        }
        else
        {
            selection->volume_instance_ids = true;
        }
    }
}

// Convert volume sets to the correct type for SensDetInserter
auto make_set_lv(inp::GeantSd::VariantSetVolume const& sv)
{
    using InpSD = inp::GeantSd;
    using SetLV = SensDetInserter::SetLV;
    return std::visit(
        Overload{[](InpSD::SetVolume const& s) {
                     return SetLV{s.begin(), s.end()};
                 },
                 [](InpSD::SetString const& s) {
                     return find_geant_volumes({s.begin(), s.end()});
                 }},
        sv);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Map detector IDs on construction.
 */
GeantSd::GeantSd(SPConstGeo geo,
                 ParticleParams const& par,
                 Input const& setup,
                 StreamId::size_type num_streams)
    : nonzero_energy_deposition_(setup.ignore_zero_deposition)
    , geo_{std::move(geo)}
{
    CELER_EXPECT(num_streams > 0);

    // Convert setup options to step data
    selection_.particle = setup.track;
    selection_.energy_deposition = setup.energy_deposition;
    selection_.step_length = setup.step_length;
    for (auto p : range(StepPoint::size_))
    {
        update_selection(&selection_.points[p], setup.points[p]);
        if (setup.points[p].touchable)
        {
            locate_touchable_[p] = true;
        }
    }
    // Hit processors MUST be allocated on the thread they're used because of
    // geant4 thread-local SDs. They MUST also be DEallocated on the same
    // thread they're created due to Geant4 thread-local allocators.
    // There must be one hit processor per thread.
    processor_weakptrs_.resize(num_streams);
    processors_.resize(num_streams);

    // Map detector volumes
    this->setup_volumes(*geo_, setup);

    if (setup.track)
    {
        this->setup_particles(par);
    }

    CELER_ENSURE(setup.track == !this->particles_.empty());
    CELER_ENSURE(geant_vols_ && geant_vols_->size() == celer_vols_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Create local hit processor.
 *
 * Due to Geant4 multithread semantics, this \b must be done on the same CPU
 * thread on which the resulting processor used. It must be done once per
 * thread and can be done separately.
 */
auto GeantSd::make_local_processor(StreamId sid) -> SPProcessor
{
    CELER_EXPECT(sid < processor_weakptrs_.size());
    CELER_EXPECT(!processors_[sid.get()]);

    auto result = std::make_shared<HitProcessor>(
        geant_vols_, geo_, particles_, selection_, locate_touchable_);
    processor_weakptrs_[sid.get()] = result;
    processors_[sid.get()] = result.get();
    return result;
}

//---------------------------------------------------------------------------//
//! Default destructor
GeantSd::~GeantSd() = default;

//---------------------------------------------------------------------------//
/*!
 * Map volume names to detector IDs and exclude tracks with no deposition.
 */
auto GeantSd::filters() const -> Filters
{
    Filters result;

    for (auto didx : range<DetectorId::size_type>(celer_vols_.size()))
    {
        result.detectors[celer_vols_[didx]] = DetectorId{didx};
    }

    result.nonzero_energy_deposition = nonzero_energy_deposition_;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void GeantSd::process_steps(HostStepState state)
{
    auto& process_hits = this->get_local_hit_processor(state.stream_id);
    process_hits(state.steps);
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void GeantSd::process_steps(DeviceStepState state)
{
    auto& process_hits = this->get_local_hit_processor(state.stream_id);
    process_hits(state.steps);
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
void GeantSd::setup_volumes(GeoParams const& geo, inp::GeantSd const& setup)
{
    // Convert labels or other set types
    auto skip_volumes = make_set_lv(setup.skip_volumes);
    auto force_volumes = make_set_lv(setup.force_volumes);

    // Helper for inserting volumes
    SensDetInserter::MapIdLv found_id_lv;
    SensDetInserter::VecLV missing_lv;
    SensDetInserter insert_volume(geo, skip_volumes, &found_id_lv, &missing_lv);

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

    // Loop over ext-specified G4LV
    for (G4LogicalVolume const* lv : force_volumes)
    {
        insert_volume(lv);
    }

    CELER_VALIDATE(
        missing_lv.empty(),
        << "failed to find unique " << cmake::core_geo
        << " volume(s) corresponding to Geant4 volume(s) "
        << join_stream(missing_lv.begin(),
                       missing_lv.end(),
                       ", ",
                       [](std::ostream& os, G4LogicalVolume const* lv) {
                           os << '"' << lv->GetName() << '"';
                       })
        << " while mapping sensitive detectors");
    CELER_VALIDATE(
        !found_id_lv.empty(),
        << R"(no G4 sensitive detectors are defined: set `SetupOptions.sd.enabled` to `false` if this is expected)");

    // Unfold map into LV/ID vectors
    VecLV geant_vols;
    geant_vols.reserve(found_id_lv.size());
    celer_vols_.reserve(found_id_lv.size());
    for (auto&& [id, lv] : found_id_lv)
    {
        geant_vols.push_back(lv);
        celer_vols_.push_back(id);
    }
    geant_vols_ = std::make_shared<VecLV>(std::move(geant_vols));
}

//---------------------------------------------------------------------------//
void GeantSd::setup_particles(ParticleParams const& par)
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
 * Return the local hit processor.
 */
auto GeantSd::get_local_hit_processor(StreamId sid) -> HitProcessor&
{
    CELER_EXPECT(sid < processors_.size());
    CELER_EXPECT(([&] {
        // Check that shared pointer is still alive
        auto sp = processor_weakptrs_[sid.get()].lock();
        return sp && sp.get() == processors_[sid.get()];
    }()));

    return *processors_[sid.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
