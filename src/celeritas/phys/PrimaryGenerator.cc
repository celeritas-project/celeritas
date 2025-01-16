//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "PrimaryGenerator.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "celeritas/Units.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/random/distribution/DeltaDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

#include "ParticleParams.hh"
#include "Primary.hh"

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the energy.
 */
PrimaryGenerator::EnergySampler
make_energy_sampler(inp::EnergyDistribution const& i)
{
    CELER_VALIDATE(i.energy > zero_quantity(),
                   << "invalid primary generator energy " << i.energy.value());

    return DeltaDistribution<real_type>(i.energy.value());
}

//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the position.
 */
auto make_position_sampler(inp::ShapeDistribution const& i)
{
    CELER_ASSUME(!i.valueless_by_exception());
    return std::visit(
        return_as<PrimaryGenerator::PositionSampler>(Overload{
            [](inp::PointShape const& ps) { return DeltaDistribution{ps.pos}; },
            [](inp::UniformBoxShape const& ubs) {
                return UniformBoxDistribution{ubs.lower, ubs.upper};
            }}),
        i);
}

//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the direction.
 */
auto make_direction_sampler(inp::AngleDistribution const& i)
{
    CELER_ASSUME(!i.valueless_by_exception());
    return std::visit(return_as<PrimaryGenerator::DirectionSampler>(Overload{
                          [](inp::IsotropicAngle const&) {
                              return IsotropicDistribution<real_type>{};
                          },
                          [](inp::MonodirectionalAngle const& ma) {
                              CELER_VALIDATE(is_soft_unit_vector(ma.dir),
                                             << "primary generator angle is "
                                                "not a unit vector");
                              return DeltaDistribution{ma.dir};
                          }}),
                      i);
}

//---------------------------------------------------------------------------//
}  // namespace

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

    return PrimaryGenerator(to_input(opts), *particles);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with options and shared particle data.
 */
PrimaryGenerator::PrimaryGenerator(Input const& i,
                                   ParticleParams const& particles)
    : num_events_{i.num_events}
    , primaries_per_event_{i.primaries_per_event}
    , seed_{i.seed}
    , sample_energy_{make_energy_sampler(i.energy)}
    , sample_pos_{make_position_sampler(i.shape)}
    , sample_dir_{make_direction_sampler(i.angle)}
{
    // TODO: seed based on event
    this->seed(UniqueEventId{0});

    particle_id_.reserve(i.pdg.size());
    for (auto const& pdg : i.pdg)
    {
        particle_id_.push_back(particles.find(pdg));
    }
    CELER_VALIDATE(
        std::all_of(particle_id_.begin(), particle_id_.end(), LogicalTrue{}),
        << R"(invalid or missing particle types specified for primary generator)");
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
        p.particle_id = particle_id_[i % particle_id_.size()];
        p.energy = units::MevEnergy{sample_energy_(rng_)};
        p.position = sample_pos_(rng_);
        p.direction = sample_dir_(rng_);
        p.time = 0;
        p.event_id = EventId{event_count_};
    }
    ++event_count_;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reseed RNG for interaction with celer-g4.
 */
void PrimaryGenerator::seed(UniqueEventId uid)
{
    CELER_EXPECT(uid);
    rng_.seed(seed_ + uid.unchecked_get());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
