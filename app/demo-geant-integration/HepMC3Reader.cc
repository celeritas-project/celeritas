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

    auto primaries = this->load_primaries();
    CELER_ASSERT(!primaries.empty());

    // Add primaries to event
    for (const auto& primary : primaries)
    {
        // TODO: Do we need to check if vertex is inside world volume?

        // All primaries start at t = 0
        auto g4_vtx = std::make_unique<G4PrimaryVertex>(
            primary.vertex[0], primary.vertex[1], primary.vertex[2], 0);

        g4_vtx->SetPrimary(new G4PrimaryParticle(primary.pdg,
                                                 primary.momentum[0],
                                                 primary.momentum[1],
                                                 primary.momentum[2],
                                                 primary.energy));

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
    num_events_          = -1;
    while (!temp_file->failed())
    {
        // Count event and try to read the next
        HepMC3::GenEvent gen_event;
        temp_file->read_event(gen_event);
        num_events_++;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
HepMC3Reader::~HepMC3Reader() = default;

//---------------------------------------------------------------------------//
/*!
 * Read event and return vector of primaries.
 */
std::vector<HepMC3Reader::Primary> HepMC3Reader::load_primaries()
{
    HepMC3::GenEvent gen_event;
    input_file_->read_event(gen_event);

    std::vector<Primary> primaries;

    if (input_file_->failed())
    {
        // End of file; return empty vector
        return primaries;
    }

    CELER_LOG_LOCAL(status)
        << "Reading HepMC3 event " << gen_event.event_number();
    CELER_EXPECT(gen_event.momentum_unit() == HepMC3::Units::MEV
                 && gen_event.length_unit() == HepMC3::Units::CM);

    const auto& pos       = gen_event.event_pos();
    const auto& particles = gen_event.particles();

    // Populate vector of primaries
    for (const auto& particle : particles)
    {
        const auto& data = particle->data();
        const auto& p    = data.momentum;

        Primary primary;
        primary.pdg    = data.pid;
        primary.energy = data.momentum.e(); // Must be in MeV
        primary.momentum.set(p.x(), p.y(), p.z());
        // Geant4 base unit is mm and thus needs to be converted
        primary.vertex.set(pos.x() * cm, pos.y() * cm, pos.z() * cm);

        primaries.push_back(std::move(primary));
    }

    return primaries;
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
