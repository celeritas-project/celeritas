//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/ExponentialDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from an exponential distribution: -log(xi) / lambda.
 *
 * This is simply an implementation of std::exponential_distribution that uses
 * the Celeritas canonical generator and is independent of library
 * implementation.
 *
 * Note (for performance-critical sections of code) that if this class is
 * constructed locally with the default value of lambda = 1.0, the
 * inversion and multiplication will be optimized out (and the code will be
 * exactly identical to `-std::log(rng.ran())`.
 */
template<class RealType = ::celeritas::real_type>
class ExponentialDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = RealType;
    //!@}

  public:
    // Construct with defaults
    explicit inline CELER_FUNCTION
    ExponentialDistribution(real_type lambda = 1);

    // Sample using the given random number generator
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& g);

  private:
    result_type neg_inv_lambda_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the mean of the exponential distribution.
 */
template<class RT>
CELER_FUNCTION
ExponentialDistribution<RT>::ExponentialDistribution(real_type lambda)
    : neg_inv_lambda_(real_type{-1} / lambda)
{
    CELER_EXPECT(lambda > real_type{0});
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
}  // namespace celeritas
