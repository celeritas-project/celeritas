//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include <set>
#include <HepMC3/GenEvent.h>
#include <HepMC3/Reader.h>
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
EventReader::EventReader(std::string const& filename,
                         SPConstParticles particles)
    : particles_(std::move(particles))
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

    // Determine the input file format and construct the appropriate reader
    reader_ = open_hepmc3(filename);

    CELER_LOG(debug) << "Reader type: "
                     << TypeDemangler<HepMC3::Reader>()(*reader_);

    CELER_ENSURE(reader_);
}

//---------------------------------------------------------------------------//
/*!
 * Read a single event from the event record.
 *
 * \note
 * Units are converted manually from HepMC3 to Celeritas rather than using \c
 * GenEvent::set_units(). This method converts the momentum units of all
 * particles and the length units of all daughter vertices; however, it does
 * *not* convert the length units of the root vertex. This means any particle
 * without a vertex will not have the correct units for position and time. This
 * may also be true for any vertices that do not have a position assigned, as
 * their positions will be inherited from the vertices of their ancestors (or
 * by falling back on the event position).
 */
auto EventReader::operator()() -> result_type
{
    CELER_EXPECT(particles_);

    // Parse the next event from the record
    HepMC3::GenEvent evt;
    {
        ScopedTimeAndRedirect temp_{"HepMC3"};
        reader_->read_event(evt);
    }
    // There are no more events
    if (reader_->failed())
    {
        return {};
    }

    CELER_LOG(debug) << "Reading event " << event_count_;
    EventId const event_id{event_count_++};
    if (id_cast<EventId>(evt.event_number()) != event_id)
    {
        CELER_LOG_LOCAL(warning)
            << "Overwriting event ID " << evt.event_number()
            << " from file with sequential event ID " << event_id.get();
    }

    std::set<int> missing_pdg;

    result_type result;
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
        ParticleId particle_id{particles_->find(pdg)};
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

        // Get the position of the vertex
        auto pos = par->production_vertex()->position();
        HepMC3::Units::convert(pos, evt.length_unit(), HepMC3::Units::CM);
        auto to_cm = [](double v) {
            return static_cast<real_type>(v) * units::centimeter;
        };
        primary.position = {to_cm(pos.x()), to_cm(pos.y()), to_cm(pos.z())};

        // Get the lab-frame time [time]
        primary.time = to_cm(pos.t()) / constants::c_light;

        // Get the direction of the primary
        auto mom = par->momentum();
        HepMC3::Units::convert(mom, evt.momentum_unit(), HepMC3::Units::MEV);
        primary.direction
            = make_unit_vector(Real3{static_cast<real_type>(mom.px()),
                                     static_cast<real_type>(mom.py()),
                                     static_cast<real_type>(mom.pz())});

        // Get the kinetic energy of the primary
        primary.energy = units::MevEnergy{static_cast<real_type>(mom.e())
                                          - static_cast<real_type>(mom.m())};

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
