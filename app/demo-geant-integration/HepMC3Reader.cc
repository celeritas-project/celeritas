//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HepMC3Reader.cc
//---------------------------------------------------------------------------//
#include "HepMC3Reader.hh"

#include <mutex>
#include <G4SystemOfUnits.hh>
#include <G4TransportationManager.hh>
#include <HepMC3/ReaderFactory.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Constants.hh"

#include "GlobalSetup.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Return non-owning pointer to a singleton.
 */
HepMC3Reader* HepMC3Reader::instance()
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
    const auto inside_status
        = G4TransportationManager::GetTransportationManager()
              ->GetNavigatorForTracking()
              ->GetWorldVolume()
              ->GetLogicalVolume()
              ->GetSolid()
              ->Inside(
                  G4ThreeVector{event_pos.x(), event_pos.y(), event_pos.z()});

    CELER_ASSERT(inside_status == EInside::kInside);

    // Add primaries to event
    const auto& gen_particles = gen_event.particles();
    for (const auto& gen_particle : gen_particles)
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
            event_pos.t() * CLHEP::s / celeritas::constants::c_light); // [s]

        const auto& p = part_data.momentum;
        g4_vtx->SetPrimary(
            new G4PrimaryParticle(part_data.pid, p.x(), p.y(), p.z(), p.e()));

        g4_event->AddPrimaryVertex(g4_vtx.release());
    }

    CELER_ENSURE(g4_event->GetNumberOfPrimaryVertex() > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with HepMC3 input filename and store total number of events.
 */
HepMC3Reader::HepMC3Reader() : G4VPrimaryGenerator()
{
    std::string filename = GlobalSetup::Instance()->GetHepmc3File();
    CELER_LOG(info) << "Constructing HepMC3 reader with " << filename;
    input_file_ = HepMC3::deduce_reader(filename);

    // Fetch total number of events
    const auto temp_file = HepMC3::deduce_reader(filename);
    num_events_          = 0;
    while (!temp_file->failed())
    {
        temp_file->skip(1);
        num_events_++;
    }

    CELER_LOG(status) << "counted events is " << num_events_;
    CELER_ENSURE(num_events_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
HepMC3Reader::~HepMC3Reader() = default;

//---------------------------------------------------------------------------//
} // namespace demo_geant
