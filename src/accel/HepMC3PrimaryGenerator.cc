//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "HepMC3PrimaryGenerator.hh"

#include <cmath>
#include <mutex>
#include <G4PhysicalConstants.hh>
#include <G4TransportationManager.hh>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenVertex.h>
#include <HepMC3/Reader.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/Types.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Add particles to an event using vertices
class PrimaryInserter
{
  public:
    explicit PrimaryInserter(G4Event* event) : g4_event_(event)
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

        // Insert primary
        auto const& p = par.momentum();
        CELER_ASSERT(g4_vtx_);
        g4_vtx_->SetPrimary(
            new G4PrimaryParticle(par.pid(), p.x(), p.y(), p.z(), p.e()));
    }

    void operator()() { this->insert_vertex(); }

  private:
    G4Event* g4_event_;
    std::unique_ptr<G4PrimaryVertex> g4_vtx_;
    HepMC3::GenVertex const* last_vtx_ = nullptr;

    void insert_vertex()
    {
        if (g4_vtx_->GetNumberOfParticle() == 0)
            return;

        auto const& pos = last_vtx_->position();
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
    HepMC3::GenEvent evt;

    {
        // Read the next event from the file.
        std::lock_guard scoped_lock{read_mutex_};
        reader_->read_event(evt);
        CELER_ASSERT(!reader_->failed());
        CELER_ASSERT(evt.particles().size() > 0);
    }

    CELER_LOG_LOCAL(debug) << "Processing " << evt.vertices().size()
                           << " vertices with " << evt.particles().size()
                           << " primaries from HepMC event ID "
                           << evt.event_number();
    if (evt.event_number() != g4_event->GetEventID())
    {
        CELER_LOG_LOCAL(warning)
            << "Read event ID " << evt.event_number()
            << " does not match Geant4 event ID " << g4_event->GetEventID();
    }

    evt.set_units(HepMC3::Units::MEV, HepMC3::Units::MM);  // Geant4 units

    int num_primaries{0};
    PrimaryInserter insert_primary{g4_event};

    for (auto const& par : evt.particles())
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

    CELER_LOG_LOCAL(info) << "Read " << g4_event->GetNumberOfPrimaryVertex()
                          << " real vertices with " << num_primaries
                          << " real primaries from HepMC event ID "
                          << evt.event_number();

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
                   << "event " << evt.event_number()
                   << " did not contain any primaries suitable for "
                      "simulation");
}

//---------------------------------------------------------------------------//
/*!
 * Copy a celeritas::Real3 to an std::array<double, 3>.
 */
void real3_to_array(Real3 const& src, std::array<double, 3>& dst)
{
    std::memcpy(&dst, &src, sizeof(src));
}

//---------------------------------------------------------------------------//
/*!
 * Dump list of primaries from HepMC3 to ROOT.
 */
void HepMC3PrimaryGenerator::dump_to_root(
    std::string const& hepmc3_input_filename,
    std::string const& root_output_filename)
{
    using std::pow;
    using std::sqrt;

    RootFileManager root_mgr(root_output_filename.c_str());
    auto tree = root_mgr.make_tree("primaries", "primaries");

    struct Primary
    {
        std::size_t event_id;
        int particle;
        double energy;
        double time;
        std::array<double, 3> pos;
        std::array<double, 3> dir;
    } prim;

    tree->Branch("event_id", &prim.event_id);
    tree->Branch("energy", &prim.energy);
    tree->Branch("time", &prim.time);
    tree->Branch("pos", &prim.pos);
    tree->Branch("dir", &prim.dir);

    EventReader reader(hepmc3_input_filename);
    auto primaries = reader();

    while (!primaries.empty())
    {
        for (auto const& primary : primaries)
        {
            prim.event_id = primary.event_id.unchecked_get();
            prim.energy = primary.energy.value();
            prim.time = primary.time;
            real3_to_array(primary.position, prim.pos);
            real3_to_array(primary.direction, prim.dir);
        }
        tree->Fill();
        primaries = reader();
    }

    // TTree and TFile write, and TFile close happen via destructors

    CELER_LOG(status) << "Generated \'" << root_output_filename
                      << "\' with list of primaries from \'"
                      << hepmc3_input_filename << "\'";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
