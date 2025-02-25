//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "HepMC3PrimaryGenerator.hh"

#include <mutex>
#include <G4LogicalVolume.hh>
#include <G4PhysicalConstants.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenVertex.h>
#include <HepMC3/Reader.h>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/io/EventReader.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Add particles to an event using vertices
class PrimaryInserter
{
  public:
    explicit PrimaryInserter(G4Event* event, HepMC3::GenEvent const& evt)
        : g4_event_(event)
        , length_unit_(evt.length_unit())
        , momentum_unit_(evt.momentum_unit())
    {
        CELER_EXPECT(g4_event_);
        g4_vtx_ = std::make_unique<G4PrimaryVertex>();
    }

    void operator()(HepMC3::GenParticle const& par)
    {
        auto* cur_vtx = par.production_vertex().get();
        if (last_vtx_ && last_vtx_ != cur_vtx)
        {
            this->insert_vertex();
        }
        last_vtx_ = cur_vtx;

        // Get the four momentum
        auto p = par.momentum();
        HepMC3::Units::convert(p, momentum_unit_, HepMC3::Units::MEV);

        // Create the primary particle and set the PDG mass. If the particle is
        // not in the \c G4ParticleTable, the mass is set to -1.  Calling the
        // constructor with the four momentum would set the mass based on the
        // relativistic energy-momentum relation.
        G4PrimaryParticle* primary = new G4PrimaryParticle(par.pid());

        // Set the primary directlon
        auto dir = make_unit_vector(Array<double, 3>{p.x(), p.y(), p.z()});
        primary->SetMomentumDirection(convert_to_geant(dir, 1));

        // Set the kinetic energy
        primary->SetKineticEnergy(p.e() - p.m());

        // Insert primary
        CELER_ASSERT(g4_vtx_);
        g4_vtx_->SetPrimary(primary);
    }

    void operator()() { this->insert_vertex(); }

  private:
    G4Event* g4_event_;
    HepMC3::Units::LengthUnit length_unit_;
    HepMC3::Units::MomentumUnit momentum_unit_;
    std::unique_ptr<G4PrimaryVertex> g4_vtx_;
    HepMC3::GenVertex const* last_vtx_ = nullptr;

    void insert_vertex()
    {
        if (g4_vtx_->GetNumberOfParticle() == 0)
            return;

        auto pos = last_vtx_->position();
        HepMC3::Units::convert(pos, length_unit_, HepMC3::Units::MM);
        g4_vtx_->SetPosition(
            pos.x() * CLHEP::mm, pos.y() * CLHEP::mm, pos.z() * CLHEP::mm);
        g4_vtx_->SetT0(pos.t() / (CLHEP::mm * CLHEP::c_light));
        g4_event_->AddPrimaryVertex(g4_vtx_.release());
        g4_vtx_ = std::make_unique<G4PrimaryVertex>();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Get the world solid volume.
 *
 * This must be called *after* detector setup, otherwise the app will crash.
 */
G4VSolid* get_world_solid()
{
    auto* world = geant_world_volume();
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
{
    // Fetch total number of events by opening a temporary reader
    num_events_ = [&filename] {
        SPReader temp_reader = open_hepmc3(filename);
        CELER_ASSERT(temp_reader);
        size_type result = 0;
#if HEPMC3_VERSION_CODE < 3002000
        HepMC3::GenEvent evt;
        temp_reader->read_event(evt);
#else
        temp_reader->skip(0);
#endif
        CELER_VALIDATE(!temp_reader->failed(),
                       << "event file '" << filename
                       << "' did not contain any events");
        do
        {
            result++;
#if HEPMC3_VERSION_CODE < 3002000
            temp_reader->read_event(evt);
#else
            temp_reader->skip(1);
#endif
        } while (!temp_reader->failed());
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
 * G4VUserPrimaryGeneratorAction::GeneratePrimaries .
 */
void HepMC3PrimaryGenerator::GeneratePrimaryVertex(G4Event* g4_event)
{
    CELER_EXPECT(g4_event && g4_event->GetEventID() >= 0);
    SPHepEvt evt
        = this->read_event(static_cast<size_type>(g4_event->GetEventID()));
    CELER_ASSERT(evt && evt->particles().size() > 0);
    CELER_LOG_LOCAL(debug) << "Processing " << evt->vertices().size()
                           << " vertices with " << evt->particles().size()
                           << " primaries from HepMC event ID "
                           << evt->event_number() << " into Geant4 event ID "
                           << g4_event->GetEventID();

    int num_primaries{0};
    PrimaryInserter insert_primary{g4_event, *evt};

    for (auto const& par : evt->particles())
    {
        if (par->data().status != 1 || par->end_vertex())
        {
            // Skip particles that should not be tracked: Geant4 HepMCEx01
            // skips all that don't have the status code of "final" (see
            // http://hepmc.web.cern.ch/hepmc/releases/HepMC2_user_manual.pdf
            // ) and furthermore skips particles that are not leaves on the
            // tree of generated particles
            continue;
        }
        ++num_primaries;
        insert_primary(*par);
    }
    insert_primary();

    CELER_LOG_LOCAL(info) << "Loaded " << g4_event->GetNumberOfPrimaryVertex()
                          << " real vertices with " << num_primaries
                          << " real primaries for event "
                          << g4_event->GetEventID();

    // Check world solid
    if (CELERITAS_DEBUG)
    {
        if (CELER_UNLIKELY(!world_solid_))
        {
            world_solid_ = get_world_solid();
        }
        CELER_ASSERT(world_solid_);
        for (auto vtx_id : range(g4_event->GetNumberOfPrimaryVertex()))
        {
            G4PrimaryVertex* vtx = g4_event->GetPrimaryVertex(vtx_id);
            CELER_ASSERT(vtx);
            CELER_ASSERT(world_solid_->Inside(vtx->GetPosition())
                         == EInside::kInside);
        }
    }

    CELER_VALIDATE(g4_event->GetNumberOfPrimaryVertex() > 0,
                   << "event " << g4_event->GetEventID()
                   << " did not contain any primaries suitable for "
                      "simulation");
}

//---------------------------------------------------------------------------//
/*!
 * Read the given event from the file in a thread-safe manner.
 *
 * Each event can only be read once. Because reading across threads may be out
 * of order, the next event to read may not be the next event in the file. To
 * fix this with minimal performance and memory impact, we read all events up
 * to the one requested into a buffer. Once the events are buffered, we release
 * the shared pointer (marking its location in the buffer as empty) and return
 * it to the calling thread. Before reading new events, empty elements at the
 * front of the buffer are released. In the usual case, the buffer should only
 * be size(num_threads), but in the worst case (the first event is very slow
 * and the other threads keep processing new events) it can be arbitrarily
 * large. However, since accessing an element in a deque is a constant-time
 * operation, this function should be constant time at best and scale with the
 * number of threads at worst.
 */
auto HepMC3PrimaryGenerator::read_event(size_type event_id) -> SPHepEvt
{
    CELER_EXPECT(event_id < num_events_);

    std::lock_guard scoped_lock{read_mutex_};
    CELER_EXPECT(event_id >= start_event_);

    // Remove empties at the front of the deque
    while (!event_buffer_.empty() && !event_buffer_.front())
    {
        event_buffer_.pop_front();
        ++start_event_;
    }

    CELER_LOG_LOCAL(debug) << "Reading to event " << event_id
                           << ": buffer has [" << start_event_ << ", "
                           << start_event_ + event_buffer_.size() << ")";

    // Read new events until we get to the requested one
    while (event_id >= start_event_ + event_buffer_.size())
    {
        size_type expected_id = start_event_ + event_buffer_.size();
        event_buffer_.push_back(std::make_shared<HepMC3::GenEvent>());
        reader_->read_event(*event_buffer_.back());

        auto read_evt_id = event_buffer_.back()->event_number();
        CELER_VALIDATE(!reader_->failed(),
                       << "event " << expected_id << " could not be read");

        if (static_cast<size_type>(read_evt_id) != expected_id)
        {
            CELER_LOG(warning)
                << "HepMC3 event IDs are not consecutive from zero: read ID "
                << read_evt_id << " but expected ID " << expected_id;
        }
    }

    // Get the event at the requested ID (if two threads erroneously requested
    // the same event, the shared pointer will be false).
    CELER_ASSERT(event_id >= start_event_
                 && event_id < start_event_ + event_buffer_.size());
    auto evt = std::move(event_buffer_[event_id - start_event_]);
    CELER_ENSURE(evt);
    return evt;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
