//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include <set>
#include <HepMC3/GenEvent.h>
#include <HepMC3/ReaderFactory.h>
#include <HepMC3/Setup.h>

#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a filename.
 */
EventReader::EventReader(std::string const& filename, SPConstParticles params)
    : params_(std::move(params))
{
    CELER_EXPECT(params_);

    // Determine the input file format and construct the appropriate reader
    reader_ = open_hepmc3(filename);

    CELER_LOG(debug) << "Reader type: "
                     << TypeDemangler<HepMC3::Reader>()(*reader_);

    CELER_ENSURE(reader_);
}

//---------------------------------------------------------------------------//
/*!
 * Read a single event from the event record.
 */
auto EventReader::operator()() -> result_type
{
    // Parse the next event from the record
    HepMC3::GenEvent evt;
    {
        CELER_LOG(debug) << "Reading event " << event_count_;
        ScopedTimeAndRedirect temp_{"HepMC3"};
        reader_->read_event(evt);
    }
    // There are no more events
    if (reader_->failed())
    {
        return {};
    }

    EventId const event_id{event_count_++};
    if (static_cast<EventId::size_type>(evt.event_number()) != event_id.get())
    {
        CELER_LOG_LOCAL(warning)
            << "Overwriting event ID " << evt.event_number()
            << " from file with sequential event ID " << event_id.get();
    }

    // Convert the energy units to MeV and the length units to cm
    evt.set_units(HepMC3::Units::MEV, HepMC3::Units::CM);

    std::set<int> missing_pdg;

    result_type result;
    int track_id = 0;
    for (auto const& par : evt.particles())
    {
        if (par->data().status != 1 || par->end_vertex())
        {
            // Skip particles that aren't leaves on the tree of generated
            // particles
            continue;
        }

        // Get the PDG code and check if this particle type is defined for
        // the current physics
        PDGNumber pdg{par->pid()};
        ParticleId particle_id{params_->find(pdg)};
        if (CELER_UNLIKELY(!particle_id))
        {
            missing_pdg.insert(pdg.unchecked_get());
            continue;
        }

        Primary primary;

        // Set the registered ID of the particle
        primary.particle_id = particle_id;

        // Set the event and track number
        primary.event_id = event_id;
        primary.track_id = TrackId(track_id++);

        // Get the position of the vertex
        auto const& pos = par->production_vertex()->position();
        primary.position = {pos.x() * units::centimeter,
                            pos.y() * units::centimeter,
                            pos.z() * units::centimeter};

        // Get the lab-frame time [s]
        primary.time = pos.t() * units::centimeter / constants::c_light;

        // Get the direction of the primary
        primary.direction = {
            par->momentum().px(), par->momentum().py(), par->momentum().pz()};
        normalize_direction(&primary.direction);

        // Get the energy of the primary
        primary.energy = units::MevEnergy{par->momentum().e()};

        result.push_back(primary);
    }

    CELER_VALIDATE(!result.empty(),
                   << "event " << event_id.get()
                   << " did not contain any primaries suitable for "
                      "simulation");

    CELER_VALIDATE(missing_pdg.empty(),
                   << "event " << event_id.get()
                   << " contains unknown particle types: PDG "
                   << join(missing_pdg.begin(), missing_pdg.end(), ", "));

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set HepMC3 verbosity from the environment.
 *
 * The default debug level is 5.
 */
void set_hepmc3_verbosity_from_env()
{
    std::string const& var = celeritas::getenv("HEPMC3_VERBOSE");
    if (!var.empty())
    {
        HepMC3::Setup::set_debug_level(std::stoi(var));
    }
    else
    {
        HepMC3::Setup::set_debug_level(1);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Wrapper function for HepMC3::deduce_reader to avoid duplicate symbols.
 *
 * HepMC3 through 3.2.6 has a ReaderFactory.h that includes function
 * *definitions* without \c inline keywords, leading to duplicate symbols.
 * Reusing this function rather than including ReaderFactory multiple times in
 * Celeritas is the easiest way to work around the problem.
 *
 * It also sets the debug level from the environment, prints a status
 * message,and validates the file.
 */
std::shared_ptr<HepMC3::Reader> open_hepmc3(std::string const& filename)
{
    set_hepmc3_verbosity_from_env();

    CELER_LOG(info) << "Opening HepMC3 input file at " << filename;

    ScopedTimeAndRedirect temp_{"HepMC3"};
    auto result = HepMC3::deduce_reader(filename);
    CELER_VALIDATE(result, << "failed to deduce event input file type");
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
