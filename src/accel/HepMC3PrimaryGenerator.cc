//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "HepMC3PrimaryGenerator.hh"

#include <mutex>
#include <G4PhysicalConstants.hh>
#include <G4TransportationManager.hh>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/Reader.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/EventReader.hh"

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
                      "HepMC3PrimaryGenerator was instantiated");
    auto* lv = world->GetLogicalVolume();
    CELER_ASSERT(lv);
    auto* solid = lv->GetSolid();
    CELER_ENSURE(solid);
    return solid;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a path to a HepMC3-compatible input file.
 */
HepMC3PrimaryGenerator::HepMC3PrimaryGenerator(std::string const& filename)
    : world_solid_{get_world_solid()}
{
    // Fetch total number of events by opening a temporary reader
    num_events_ = [&filename] {
        SPReader temp_reader = open_hepmc3(filename);
        CELER_ASSERT(temp_reader);
        int result = 0;
        while (!temp_reader->failed())
        {
            temp_reader->skip(1);
            result++;
        }
        CELER_LOG(debug) << "HepMC3 file has " << result << " events";
        return result;
    }();

    // Open a persistent reader
    reader_ = open_hepmc3(filename);

    CELER_ENSURE(reader_);
    CELER_ENSURE(num_events_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Add HepMC3 primaries to a Geant4 event.
 *
 * This function should be called by \c
 * G4VUserPrimaryGeneratorAction::GeneratePrimaries . It is thread safe as long
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
void HepMC3PrimaryGenerator::GeneratePrimaryVertex(G4Event* g4_event)
{
    HepMC3::GenEvent gen_event;

    {
        // Read the next event from the file.
        std::lock_guard scoped_lock{read_mutex_};
        reader_->read_event(gen_event);
        CELER_ASSERT(!reader_->failed());
    }

    CELER_LOG_LOCAL(info) << "Read " << gen_event.particles().size()
                          << " primaries from HepMC event ID "
                          << gen_event.event_number() << " for Geant4 event "
                          << g4_event->GetEventID();

    gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::MM);  // Geant4
                                                                 // units
    auto const& event_pos = gen_event.event_pos();

    // Verify that vertex is inside the world volume
    if (CELERITAS_DEBUG && CELER_UNLIKELY(!world_solid_))
    {
        world_solid_ = get_world_solid();
    }
    CELER_ASSERT(world_solid_->Inside(G4ThreeVector{
                     event_pos.x(), event_pos.y(), event_pos.z()})
                 == EInside::kInside);

    for (auto const& gen_particle : gen_event.particles())
    {
        // Convert primary to Geant4 vertex
        HepMC3::GenParticleData const& part_data = gen_particle->data();

        if (part_data.status <= 0)
        {
            // Skip particles that should not be tracked
            // Status codes (page 13):
            // http://hepmc.web.cern.ch/hepmc/releases/HepMC2_user_manual.pdf
            if (part_data.momentum.e() > 0)
            {
                CELER_LOG_LOCAL(debug)
                    << "Skipped status code " << part_data.status << " for "
                    << part_data.momentum.e() << " MeV primary";
            }
            continue;
        }

        auto g4_vtx = std::make_unique<G4PrimaryVertex>(
            event_pos.x(),
            event_pos.y(),
            event_pos.z(),
            event_pos.t() / CLHEP::c_light);  // [ns] (Geant4 standard unit)

        auto const& p = part_data.momentum;
        g4_vtx->SetPrimary(
            new G4PrimaryParticle(part_data.pid, p.x(), p.y(), p.z(), p.e()));

        g4_event->AddPrimaryVertex(g4_vtx.release());
    }

    CELER_ENSURE(g4_event->GetNumberOfPrimaryVertex() > 0);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
