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
    : num_events_(options.num_events)
    , primaries_per_event_(options.primaries_per_event)
{
    primary_.particle_id = particles->find(PDGNumber{options.pdg});
    primary_.energy      = units::MevEnergy{options.energy};
    primary_.position    = options.position;
    primary_.direction   = options.direction;
    primary_.time        = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Generate primary particles.
 */
auto PrimaryGenerator::operator()() -> VecPrimary
{
    VecPrimary result;
    result.reserve(num_events_ * primaries_per_event_);
    for (auto i : range(num_events_))
    {
        for (auto j : range(primaries_per_event_))
        {
            primary_.event_id = EventId{i};
            primary_.track_id = TrackId{j};
            result.push_back(primary_);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
