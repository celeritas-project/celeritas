//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "PrimaryGenerator.hh"

#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

#include "PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options and shared particle data.
 */
PrimaryGenerator::PrimaryGenerator(SPConstParticles        particles,
                                   PrimaryGeneratorOptions options)
    : particles_(std::move(particles)), options_(options)
{
}

//---------------------------------------------------------------------------//
/*!
 * Generate primary particles.
 */
auto PrimaryGenerator::operator()() -> VecPrimary
{
    Primary p;
    p.particle_id = particles_->find(PDGNumber{options_.pdg});
    p.energy      = units::MevEnergy{options_.energy};
    p.position    = options_.position;
    p.direction   = options_.direction;
    p.time        = 0;

    VecPrimary result;
    result.reserve(options_.num_events * options_.primaries_per_event);
    for (auto i : range(options_.num_events))
    {
        for (auto j : range(options_.primaries_per_event))
        {
            p.event_id = EventId{i};
            p.track_id = TrackId{j};
            result.push_back(p);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
