//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HepMC3Reader.cc
//---------------------------------------------------------------------------//
#include "HepMC3Reader.hh"

#include <mutex>
#include <G4PhysicalConstants.hh>
#include <G4TransportationManager.hh>
#include <HepMC3/ReaderFactory.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "GlobalSetup.hh"

namespace demo_geant
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
 * Return non-owning pointer to a singleton.
 */
HepMC3Reader* HepMC3Reader::Instance()
{
    static HepMC3Reader hepmc3_reader_singleton;
    return &hepmc3_reader_singleton;
}

//---------------------------------------------------------------------------//
/*!
 * Add HepMC3 primaries to a Geant4 event. This function is called by
 * `G4VUserPrimaryGeneratorAction::GeneratePrimaries`.
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
    static std::mutex           hepmc3_mutex;
    std::lock_guard<std::mutex> scoped_lock{hepmc3_mutex};

    HepMC3::GenEvent gen_event;
    input_file_->read_event(gen_event);
    CELER_ASSERT(!input_file_->failed());

    CELER_LOG_LOCAL(status)
        << "Reading HepMC3 event " << gen_event.event_number();

    gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::MM); // Geant4 units
    const auto& event_pos = gen_event.event_pos();

    // Verify if vertex is inside the world volume
    CELER_ASSERT(world_solid_->Inside(G4ThreeVector{
                     event_pos.x(), event_pos.y(), event_pos.z()})
                 == EInside::kInside);

    // Add primaries to event
    for (const auto& gen_particle : gen_event.particles())
    {
        const auto& part_data = gen_particle->data();

        if (part_data.status != 1)
        {
            // Skip particles that should not be tracked
            // Status codes (page 13):
            // http://hepmc.web.cern.ch/hepmc/releases/HepMC2_user_manual.pdf
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
/*!
 * Construct with provided HepMC3 input filename.
 */
HepMC3Reader::HepMC3Reader()
    : G4VPrimaryGenerator(), world_solid_(get_world_solid())
{
    const std::string filename = GlobalSetup::Instance()->GetHepmc3File();
    CELER_LOG(info) << "Constructing HepMC3 reader with " << filename;
    input_file_ = HepMC3::deduce_reader(filename);
    CELER_VALIDATE(input_file_, << "failed to deduce event input file type");

    // Fetch total number of events
    const auto temp_file = HepMC3::deduce_reader(filename);
    num_events_          = 0;
    while (!temp_file->failed())
    {
        temp_file->skip(1);
        num_events_++;
    }

    CELER_LOG(debug) << "HepMC3 file has " << num_events_ << " events";
    CELER_ENSURE(num_events_ > 0);
    CELER_ENSURE(world_solid_);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
HepMC3Reader::~HepMC3Reader() = default;

//---------------------------------------------------------------------------//
} // namespace demo_geant
