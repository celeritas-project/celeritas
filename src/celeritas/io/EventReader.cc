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

#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/math/ArrayUtils.hh"
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

    CELER_LOG(info) << "Opening event file at " << filename;
    ScopedTimeAndRedirect temp_{"HepMC3"};

    // Determine the input file format and construct the appropriate reader
    HepMC3::Setup::set_debug_level(1);
    reader_ = HepMC3::deduce_reader(filename);

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
    HepMC3::GenEvent gen_event;
    {
        ScopedTimeAndRedirect temp_{"HepMC3"};
        reader_->read_event(gen_event);
    }
    // There are no more events
    if (reader_->failed())
    {
        return {};
    }

    EventId const event_id{event_count_++};
    if (static_cast<EventId::size_type>(gen_event.event_number())
        != event_id.get())
    {
        CELER_LOG_LOCAL(warning)
            << "Overwriting event ID " << gen_event.event_number()
            << " from file with sequential event ID " << event_id.get();
    }

    // Convert the energy units to MeV and the length units to cm
    gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::CM);

    std::set<int> missing_pdg;

    result_type result;
    int track_id = 0;
    for (auto gen_particle : gen_event.particles())
    {
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

        // Get the PDG code and check if this particle type is defined for
        // the current physics
        PDGNumber pdg{gen_particle->pid()};
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

        // Get the position of the primary
        auto const& pos = gen_particle->production_vertex()->position();
        primary.position = {pos.x() * units::centimeter,
                            pos.y() * units::centimeter,
                            pos.z() * units::centimeter};

        // Get the lab-frame time [s]
        primary.time = pos.t() * units::centimeter / constants::c_light;

        // Get the direction of the primary
        primary.direction = {gen_particle->momentum().px(),
                             gen_particle->momentum().py(),
                             gen_particle->momentum().pz()};
        normalize_direction(&primary.direction);

        // Get the energy of the primary
        primary.energy = units::MevEnergy{gen_particle->momentum().e()};

        result.push_back(primary);
    }

    CELER_VALIDATE(missing_pdg.empty(),
                   << "Event " << event_id.get()
                   << " contains unknown particle types: PDG "
                   << join(missing_pdg.begin(), missing_pdg.end(), ", "));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
