//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/RejectionSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return whether a rejection loop needs to continue trying.
 *
 * A common implementation of sampling from a "difficult" (non-analytically
 * invertible) probability distribution function is to bound the difficult
 * distribution \em f(x) with another easily sampled function \em g(x) . Given
 a
 * maximum value \em M over the \em x interval being sampled, it is equivalent
 * to sampling \em f(x) by instead sampling from \em g(x) and rejecting with
 * probability \f[
   \frac{f(x)}{M g(x)}
 * \f]
 *
 * These invocations generate statistically equivalent results:
 *  - `BernoulliDistribution(1 - p / pmax)(rng);`
 *  - `!BernoulliDistribution(p / pmax)(rng);`
 *  - `p < pmax * UniformDistribution{}(rng)`
 *  - `RejectionSampler(p, pmax)(rng);`
 *
 * This is meant for rejection sampling, e.g., on cross section:
 * \code
    do {
      xs = sample_xs(rng);
    } while (RejectionSampler{xs, xs_max}(rng));
   \endcode
 */
template<class RealType = ::celeritas::real_type>
class RejectionSampler
{
    static_assert(std::is_floating_point_v<RealType>);

  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct with acceptance probability
    inline CELER_FUNCTION RejectionSampler(real_type f, real_type fmax);

    //! Construct when the distribution's maximum is normalized
    explicit CELER_FUNCTION RejectionSampler(real_type f)
        : RejectionSampler{f, 1}
    {
    }

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    RealType f_;
    RealType fmax_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with acceptance probability and maximum probability.
 */
template<class RealType>
CELER_FUNCTION
RejectionSampler<RealType>::RejectionSampler(real_type f, real_type fmax)
    : f_{f}, fmax_{fmax}
{
    CELER_EXPECT(f_ >= 0);
    CELER_EXPECT(fmax_ >= f_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto RejectionSampler<RealType>::operator()(Generator& rng)
    -> result_type
{
    return f_ < fmax_ * generate_canonical<RealType>(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
