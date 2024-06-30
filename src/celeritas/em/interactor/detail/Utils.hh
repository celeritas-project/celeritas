//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample azimuthal angle, calculate Cartesian vector, and rotate direction.
 */
struct CartesianTransformSampler
{
    real_type costheta;
    Real3 const& dir;

    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng)
    {
        UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
        return rotate(from_spherical(costheta, sample_phi(rng)), dir);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Calculate exiting direction via conservation of momentum.
 */
inline CELER_FUNCTION Real3 calc_exiting_direction(real_type inc_momentum,
                                                   real_type out_momentum,
                                                   Real3 const& inc_dir,
                                                   Real3 const& out_dir)
{
    Real3 result;
    for (int i = 0; i < 3; ++i)
    {
        result[i] = inc_dir[i] * inc_momentum - out_dir[i] * out_momentum;
    }
    return make_unit_vector(result);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
