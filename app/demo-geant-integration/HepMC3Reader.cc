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

namespace
{
// Mutex visible to all threads
static std::mutex mutex;
} // namespace

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Singleton declaration.
 */
static HepMC3Reader* hepmc3_reader_singleton = nullptr;

//---------------------------------------------------------------------------//
/*!
 * Return non-owning pointer to a singleton.
 */
HepMC3Reader* HepMC3Reader::instance()
{
    {
        mutex.lock();
        if (!hepmc3_reader_singleton)
        {
            hepmc3_reader_singleton = new HepMC3Reader();
        }
        mutex.unlock();
    }

    CELER_LOG_LOCAL(status) << "Instancing HepMC3 singleton";
    return hepmc3_reader_singleton;
}

//---------------------------------------------------------------------------//
/*!
 * Add HepMC3 primaries to a Geant4 event. This function is called by
 * `G4VUserPrimaryGeneratorAction::GeneratePrimaries`.
 */
void HepMC3Reader::GeneratePrimaryVertex(G4Event* g4_event)
{
    mutex.lock();

    // Populate event_primaries
    auto result = this->store_primaries();
    CELER_ASSERT(result);

    // Add primaries to event
    for (const auto& primary : event_primaries_)
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
    const int num_vertices = g4_event->GetNumberOfPrimaryVertex();
    CELER_ENSURE(num_vertices);

    mutex.unlock();
}

//---------------------------------------------------------------------------//
/*!
 * Construct with HepMC3 input filename and store total number of events.
 */
HepMC3Reader::HepMC3Reader() : G4VPrimaryGenerator()
{
    std::string filename = GlobalSetup::Instance()->GetHepmc3File();
    CELER_LOG_LOCAL(status) << "Constructing HepMC3 reader with " << filename;
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
 * Read event and populate the vector of primaries.
 */
bool HepMC3Reader::store_primaries()
{
    HepMC3::GenEvent gen_event;
    input_file_->read_event(gen_event);

    if (input_file_->failed())
    {
        // End of file
        return false;
    }

    CELER_LOG_LOCAL(status)
        << "Reading HepMC3 event " << gen_event.event_number();
    CELER_EXPECT(gen_event.momentum_unit() == HepMC3::Units::MEV
                 && gen_event.length_unit() == HepMC3::Units::CM);

    // Clear vector for new event
    event_primaries_.clear();

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

        event_primaries_.push_back(std::move(primary));
    }

    return true;
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
