//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ExponentialDistribution.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Assert.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from the mean of the exponential distribution.
 */
template<class RT>
CELER_FUNCTION
ExponentialDistribution<RT>::ExponentialDistribution(real_type lambda)
    : neg_inv_lambda_(real_type{-1} / lambda)
{
    REQUIRE(lambda > real_type{0});
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RT>
template<class Generator>
CELER_FUNCTION auto ExponentialDistribution<RT>::operator()(Generator& rng)
    -> result_type
{
    return std::log(generate_canonical<RT>(rng)) * neg_inv_lambda_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
