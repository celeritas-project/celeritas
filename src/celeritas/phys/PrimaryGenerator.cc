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
PrimaryGenerator::PrimaryGenerator(SPConstParticles               particles,
                                   const PrimaryGeneratorOptions& options)
    : num_events_(options.num_events)
    , primaries_per_event_(options.primaries_per_event)
{
    CELER_EXPECT(options);
    primary_.particle_id = particles->find(PDGNumber{options.pdg});
    primary_.energy      = units::MevEnergy{options.energy};
    primary_.position    = options.position;
    primary_.direction   = options.direction;
    primary_.time        = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Generate primary particles from a single event.
 */
auto PrimaryGenerator::operator()() -> VecPrimary
{
    if (event_count_ == num_events_)
    {
        return {};
    }

    VecPrimary result(primaries_per_event_, primary_);
    for (auto i : range(primaries_per_event_))
    {
        result[i].event_id = EventId{event_count_};
        result[i].track_id = TrackId{i};
    }
    ++event_count_;
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
