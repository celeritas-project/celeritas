//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EventReader.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include "base/ArrayUtils.hh"
#include "physics/base/Units.hh"
#include "HepMC3/GenEvent.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a filename.
 */
EventReader::EventReader(const char* filename, constSPParticleParams params)
    : params_(std::move(params))
{
    // Determine the input file format and construct the appropriate reader
    input_file_ = HepMC3::deduce_reader(filename);
    ENSURE(input_file_);
}

//---------------------------------------------------------------------------//
/*!
 * Read the primary particles from the event record.
 */
EventReader::result_type EventReader::operator()()
{
    result_type result;
    int         event_id = -1;

    while (!input_file_->failed())
    {
        // Parse the next event from the record
        HepMC3::GenEvent gen_event;
        input_file_->read_event(gen_event);

        // There are no more events
        if (input_file_->failed())
        {
            break;
        }
        ++event_id;

        // Convert the energy units to MeV and the length units to cm
        gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::CM);

        for (auto gen_particle : gen_event.particles())
        {
            // Get the PDG code and check if this particle type is defined for
            // the current physics
            PDGNumber     pdg{gen_particle->pid()};
            ParticleDefId def_id{params_->find(pdg)};
            CHECK(def_id);

            Primary primary;

            // Set the registered ID of the particle
            primary.def_id = def_id;

            // Set the event number
            primary.event_id = EventId(event_id);

            // Get the position of the primary
            auto pos         = gen_event.event_pos();
            primary.position = {pos.x() * units::centimeter,
                                pos.y() * units::centimeter,
                                pos.z() * units::centimeter};

            // Get the direction of the primary
            primary.direction = {gen_particle->momentum().px(),
                                 gen_particle->momentum().py(),
                                 gen_particle->momentum().pz()};
            normalize_direction(&primary.direction);

            // Get the energy of the primary
            primary.energy = units::MevEnergy{gen_particle->momentum().e()};

            result.push_back(primary);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
