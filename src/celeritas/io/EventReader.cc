//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include <HepMC3/GenEvent.h>
#include <HepMC3/ReaderFactory.h>

#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
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

    // Turn off HepMC3 diagnostic output that pollutes our own output
    HepMC3::Setup::set_debug_level(-1);

    // Determine the input file format and construct the appropriate reader
    input_file_ = HepMC3::deduce_reader(filename);
    CELER_ENSURE(input_file_);
}

//---------------------------------------------------------------------------//
//! Default destructor
EventReader::~EventReader() = default;

//---------------------------------------------------------------------------//
/*!
 * Read a single event from the event record.
 */
auto EventReader::operator()() -> result_type
{
    // Parse the next event from the record
    HepMC3::GenEvent gen_event;
    input_file_->read_event(gen_event);

    // There are no more events
    if (input_file_->failed())
    {
        return {};
    }

    // Convert the energy units to MeV and the length units to cm
    gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::CM);

    result_type result;
    int track_id = 0;
    for (auto gen_particle : gen_event.particles())
    {
        // Get the PDG code and check if this particle type is defined for
        // the current physics
        PDGNumber pdg{gen_particle->pid()};
        ParticleId particle_id{params_->find(pdg)};
        CELER_ASSERT(particle_id);

        Primary primary;

        // Set the registered ID of the particle
        primary.particle_id = particle_id;

        // Set the event and track number
        primary.event_id = EventId(event_count_);
        primary.track_id = TrackId(track_id++);

        // Get the position of the primary
        auto pos = gen_event.event_pos();
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
    ++event_count_;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
