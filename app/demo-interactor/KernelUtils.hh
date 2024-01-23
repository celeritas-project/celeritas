//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KernelUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/ExponentialDistribution.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

template<class Rng>
inline CELER_FUNCTION void move_to_collision(ParticleTrackView const& particle,
                                             XsCalculator const& calc_xs,
                                             Real3 const& direction,
                                             Real3* position,
                                             real_type* time,
                                             Rng& rng)
{
    CELER_EXPECT(position && time);

    // Calculate cross section at the particle's energy
    real_type sigma = calc_xs(particle.energy());
    ExponentialDistribution<real_type> sample_distance(sigma);
    // Sample distance-to-collision
    real_type distance = sample_distance(rng);
    // Move particle
    axpy(distance, direction, position);
    // Update time
    *time += distance * native_value_from(particle.speed());
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
