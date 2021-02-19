//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/ArrayUtils.hh"
#include "base/Macros.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/grid/PhysicsGridCalculator.hh"
#include "random/distributions/ExponentialDistribution.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//

template<class Rng>
inline CELER_FUNCTION void
move_to_collision(const celeritas::ParticleTrackView&     particle,
                  const celeritas::PhysicsGridCalculator& calc_xs,
                  const celeritas::Real3&                 direction,
                  celeritas::Real3*                       position,
                  celeritas::real_type*                   time,
                  Rng&                                    rng)
{
    CELER_EXPECT(position && time);
    using celeritas::real_type;

    // Calculate cross section at the particle's energy
    real_type sigma = calc_xs(particle.energy());
    celeritas::ExponentialDistribution<real_type> sample_distance(sigma);
    // Sample distance-to-collision
    real_type distance = sample_distance(rng);
    // Move particle
    axpy(distance, direction, position);
    // Update time
    *time += distance * unit_cast(particle.speed());
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
