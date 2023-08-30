//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "PrimaryGenerator.hh"

#include "corecel/cont/Range.hh"
#include "celeritas/Units.hh"
#include "celeritas/random/distribution/DeltaDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

#include "ParticleParams.hh"
#include "Primary.hh"

namespace celeritas
{
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

    using DS = DistributionSelection;

    PrimaryGenerator::Input inp;
    inp.pdg = std::move(opts.pdg);
    inp.num_events = opts.num_events;
    inp.primaries_per_event = opts.primaries_per_event;

    // Create energy distribution
    {
        auto const& p = opts.energy.params;
        switch (opts.energy.distribution)
        {
            case DS::delta:
                CELER_ASSERT(p.size() == 1);
                inp.sample_energy = DeltaDistribution<real_type>(p[0]);
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    // Create spatial distribution
    {
        auto const& p = opts.position.params;
        switch (opts.position.distribution)
        {
            case DS::delta:
                CELER_ASSERT(p.size() == 3);
                inp.sample_pos
                    = DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
                break;
            case DS::box:
                CELER_ASSERT(p.size() == 6);
                inp.sample_pos = UniformBoxDistribution<real_type>(
                    Real3{p[0], p[1], p[2]}, Real3{p[3], p[4], p[5]});
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    // Create angular distribution
    {
        auto const& p = opts.direction.params;
        switch (opts.direction.distribution)
        {
            case DS::delta:
                CELER_ASSERT(p.size() == 3);
                inp.sample_dir
                    = DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
                break;
            case DS::isotropic:
                CELER_ASSERT(p.empty());
                inp.sample_dir = IsotropicDistribution<real_type>();
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }

    return PrimaryGenerator(particles, inp);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
