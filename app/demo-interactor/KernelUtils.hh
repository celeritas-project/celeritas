//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
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

namespace demo_interactor
{
//---------------------------------------------------------------------------//

template<class Rng>
inline CELER_FUNCTION void
move_to_collision(const celeritas::ParticleTrackView& particle,
                  const celeritas::XsCalculator&      calc_xs,
                  const celeritas::Real3&             direction,
                  celeritas::Real3*                   position,
                  celeritas::real_type*               time,
                  Rng&                                rng)
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
    *time += distance * native_value_from(particle.speed());
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
