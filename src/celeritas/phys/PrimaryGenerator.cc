//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "PrimaryGenerator.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "celeritas/Units.hh"

#include "ParticleParams.hh"
#include "Primary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from user input.
 *
 * This creates a \c PrimaryGenerator from options read from a JSON input using
 * a few predefined energy, spatial, and angular distributions (that can be
 * extended as needed).
 */
PrimaryGenerator
PrimaryGenerator::from_options(SPConstParticles particles,
                               PrimaryGeneratorOptions const& opts)
{
    CELER_EXPECT(opts);

    PrimaryGenerator::Input inp;
    inp.seed = opts.seed;
    inp.pdg = std::move(opts.pdg);
    inp.num_events = opts.num_events;
    inp.primaries_per_event = opts.primaries_per_event;
    inp.sample_energy = make_energy_sampler(opts.energy);
    inp.sample_pos = make_position_sampler(opts.position);
    inp.sample_dir = make_direction_sampler(opts.direction);
    return PrimaryGenerator(particles, inp);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with options and shared particle data.
 */
PrimaryGenerator::PrimaryGenerator(SPConstParticles particles, Input const& inp)
    : num_events_(inp.num_events)
    , primaries_per_event_(inp.primaries_per_event)
    , sample_energy_(std::move(inp.sample_energy))
    , sample_pos_(std::move(inp.sample_pos))
    , sample_dir_(std::move(inp.sample_dir))
{
    CELER_EXPECT(particles);

    rng_.seed(inp.seed);
    particle_id_.reserve(inp.pdg.size());
    for (auto const& pdg : inp.pdg)
    {
        particle_id_.push_back(particles->find(pdg));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate primary particles from a single event.
 */
auto PrimaryGenerator::operator()() -> result_type
{
    if (event_count_ == num_events_)
    {
        return {};
    }

    result_type result(primaries_per_event_);
    for (auto i : range(primaries_per_event_))
    {
        Primary& p = result[i];
        p.particle_id = particle_id_[primary_count_ % particle_id_.size()];
        p.energy = units::MevEnergy{sample_energy_(rng_)};
        p.position = sample_pos_(rng_);
        p.direction = sample_dir_(rng_);
        p.time = 0;
        p.event_id = EventId{event_count_};
        p.track_id = TrackId{i};
        ++primary_count_;
    }
    ++event_count_;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
