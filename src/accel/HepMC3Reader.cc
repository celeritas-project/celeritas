//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3Reader.cc
//---------------------------------------------------------------------------//
#include "HepMC3Reader.hh"

#include <mutex>
#include <G4PhysicalConstants.hh>
#include <G4TransportationManager.hh>
#include <HepMC3/ReaderFactory.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get the world solid volume.
 *
 * This must be called *after* detector setup, otherwise the app will crash.
 */
G4VSolid* get_world_solid()
{
    auto* nav = G4TransportationManager::GetTransportationManager()
                    ->GetNavigatorForTracking();
    CELER_ASSERT(nav);
    auto* world = nav->GetWorldVolume();
    CELER_VALIDATE(world,
                   << "detector geometry was not initialized before "
                      "HepMC3Reader was instantiated");
    auto* lv = world->GetLogicalVolume();
    CELER_ASSERT(lv);
    auto* solid = lv->GetSolid();
    CELER_ENSURE(solid);
    return solid;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with provided HepMC3 input filename.
 */
HepMC3Reader::HepMC3Reader(const std::string& filename)
    : world_solid_{get_world_solid()}
{
    CELER_LOG(info) << "Loading HepMC3 input file at " << filename;
    reader_ = HepMC3::deduce_reader(filename);
    CELER_VALIDATE(reader_, << "failed to deduce event input file type");

    // Fetch total number of events
    SPReader temp_reader = HepMC3::deduce_reader(filename);
    CELER_ASSERT(temp_reader);
    num_events_ = 0;
    while (!temp_reader->failed())
    {
        temp_reader->skip(1);
        num_events_++;
    }

    CELER_LOG(debug) << "HepMC3 file has " << num_events_ << " events";
    CELER_ENSURE(num_events_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Add HepMC3 primaries to a Geant4 event.
 *
 * This function should be called by \c
 * G4VUserPrimaryGeneratorAction::GeneratePrimaries . It is thread safe as lnog
 * as \c g4_event is thread-local.
 *
 * \note
 * Current implementation is compatible with our utils/hepmc3-generator
 * (https://github.com/celeritas-project/utils) output files, including the
 * translated CMS' Pythia HEPEVT output files to HepMC3 format. Nevertheless,
 * these files do not have more complex topologies with multiple vertices with
 * mother/daughter particles. If more complex inputs are used, this will have
 * to be updated.
 */
void HepMC3Reader::GeneratePrimaryVertex(G4Event* g4_event)
{
    HepMC3::GenEvent gen_event;

    {
        // Read the next event from the file.
        G4AutoLock scoped_lock{read_mutex_};
        reader_->read_event(gen_event);
        CELER_ASSERT(!reader_->failed());
    }

    CELER_LOG_LOCAL(info) << "Read " << gen_event.particles().size()
                          << " primaries from event "
                          << gen_event.event_number();

    gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::MM); // Geant4 units
    const auto& event_pos = gen_event.event_pos();

    // Verify that vertex is inside the world volume
    if (CELERITAS_DEBUG && CELER_UNLIKELY(!world_solid_))
    {
        world_solid_ = get_world_solid();
    }
    CELER_ASSERT(world_solid_->Inside(G4ThreeVector{
                     event_pos.x(), event_pos.y(), event_pos.z()})
                 == EInside::kInside);

    for (const auto& gen_particle : gen_event.particles())
    {
        // Convert primary to Geant4 vertex
        const auto& part_data = gen_particle->data();

        if (part_data.status <= 0)
        {
            // Skip particles that should not be tracked
            // Status codes (page 13):
            // http://hepmc.web.cern.ch/hepmc/releases/HepMC2_user_manual.pdf
            CELER_LOG_LOCAL(info)
                << "Skipped status code " << part_data.status << " for "
                << part_data.momentum.e() << " MeV primary";
            continue;
        }

        auto g4_vtx = std::make_unique<G4PrimaryVertex>(
            event_pos.x(),
            event_pos.y(),
            event_pos.z(),
            event_pos.t() / CLHEP::c_light); // [ns] (Geant4 standard unit)

        const auto& p = part_data.momentum;
        g4_vtx->SetPrimary(
            new G4PrimaryParticle(part_data.pid, p.x(), p.y(), p.z(), p.e()));

        g4_event->AddPrimaryVertex(g4_vtx.release());
    }

    CELER_ENSURE(g4_event->GetNumberOfPrimaryVertex() > 0);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
