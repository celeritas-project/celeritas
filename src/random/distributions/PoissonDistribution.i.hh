//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PoissonDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from the mean of the Poisson distribution.
 */
template<class RealType>
CELER_FUNCTION
PoissonDistribution<RealType>::PoissonDistribution(real_type lambda)
    : lambda_(lambda), sample_normal_(lambda_, std::sqrt(lambda_))
{
    CELER_EXPECT(lambda_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto PoissonDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    if (lambda_ <= PoissonDistribution::lambda_threshold())
    {
        // Use direct method
        result_type k = 0;
        real_type   p = std::exp(lambda_);
        do
        {
            ++k;
            p *= generate_canonical(rng);
        } while (p > 1);
        return k - 1;
    }
    // Use Gaussian approximation rounded to nearest integer
    return result_type(sample_normal_(rng) + 0.5);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
