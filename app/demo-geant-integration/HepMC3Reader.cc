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
#include <HepMC3/ReaderFactory.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

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
    const auto& pos       = gen_event.event_pos();
    const auto& primaries = gen_event.particles();

    // Add primaries to event
    for (const auto& primary : primaries)
    {
        const auto& data = primary->data();
        const auto& p    = data.momentum;

        // All primaries start at t = 0
        auto g4_vtx
            = std::make_unique<G4PrimaryVertex>(pos.x(), pos.y(), pos.z(), 0);

        // TODO: Do we need to check if vertex is inside world volume?

        g4_vtx->SetPrimary(new G4PrimaryParticle(
            data.pid, p.x(), p.y(), p.z(), data.momentum.e()));

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
