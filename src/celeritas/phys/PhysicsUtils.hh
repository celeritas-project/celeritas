//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Particle momenntum
struct Momentum
{
    real_type magnitude;
    Real3 const& direction;
};

//---------------------------------------------------------------------------//
/*!
 * Calculate exiting direction via conservation of momentum.
 */
inline CELER_FUNCTION Real3 calc_exiting_direction(Momentum inc_momentum,
                                                   Momentum out_momentum)
{
    Real3 result;
    for (int i = 0; i < 3; ++i)
    {
        result[i] = inc_momentum.direction[i] * inc_momentum.magnitude
                    - out_momentum.direction[i] * out_momentum.magnitude;
    }
    return make_unit_vector(result);
}

//---------------------------------------------------------------------------//
/*!
 * Sample an exiting direction from a polar cosine and incident direction.
 *
 * Combine an already-sampled change in polar cosine (dot product of incident
 * and exiting) with a sampled uniform azimuthal direction, and apply that
 * rotation to the original track's incident direction.
 */
struct ExitingDirectionSampler
{
    real_type costheta;
    Real3 const& direction;

    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng)
    {
        UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
        return rotate(from_spherical(costheta, sample_phi(rng)), direction);
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
