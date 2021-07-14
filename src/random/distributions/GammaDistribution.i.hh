//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GammaDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "base/Algorithms.hh"
#include "base/Constants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from the shape and scale parameters.
 */
template<class RealType>
CELER_FUNCTION
GammaDistribution<RealType>::GammaDistribution(real_type alpha, real_type beta)
    : alpha_(alpha)
    , beta_(beta)
    , alpha_p_(alpha < 1 ? alpha + 1 : alpha)
    , d_(alpha_p_ - real_type(1) / 3)
    , c_(1 / std::sqrt(9 * d_))
{
    CELER_EXPECT(alpha_ > 0);
    CELER_EXPECT(beta_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto GammaDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    real_type u, v, z;
    do
    {
        do
        {
            z = sample_normal_(rng);
            v = 1 + c_ * z;
        } while (v <= 0);
        v = ipow<3>(v);
        u = generate_canonical(rng);
    } while (u > 1 - real_type(0.0331) * ipow<4>(z)
             && std::log(u) > real_type(0.5) * ipow<2>(z)
                                  + d_ * (1 - v + std::log(v)));

    result_type result = d_ * v * beta_;
    if (alpha_ != alpha_p_)
        result *= std::pow(generate_canonical(rng), 1 / alpha_);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
