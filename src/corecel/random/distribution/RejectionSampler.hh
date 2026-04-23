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
#include "corecel/math/NumericLimits.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return whether a rejection loop needs to continue trying.
 *
 * A common implementation of sampling from a "difficult" (non-analytically
 * invertible) probability distribution function is to bound the difficult
 * distribution \em f(x) with another easily sampled proposal distribution
 * \em g(x) .
 * Given a constant \em M such that \f$ f(x) < M g(x) \f$ over the \em x
 * interval being sampled, it is equivalent to sampling \em f(x) by instead
 * sampling from \em g(x) and rejecting with probability \f[
   \frac{f(x)}{M g(x)}
 * \f].
 *
 * This rejection sampler is constructed with the maximum \em M and is invoked
 * with the sampled ratio \f$ f(x)/g(x) \f$ and an RNG.
 * It allows an underestimate of the constant \em M by
 * \f$ 10 \epsilon_\textrm{mach} \f$ which may result in an imperceptible bias
 * against selecting the peak of the PDF.
 *
 * These invocations generate statistically equivalent results:
 *  - `BernoulliDistribution(1 - p / pmax)(rng);`
 *  - `!BernoulliDistribution(p / pmax)(rng);`
 *  - `p < pmax * UniformDistribution{}(rng)`
 *  - `RejectionSampler(pmax)(p, rng);`
 *
 * \par Example
 * This rejects over a cross section distribution with a known maximum bound.
 * \code
    RejectionSampler reject{xs_max}
    do {
      xs = sample_xs(rng);
    } while (reject(xs, rng));
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
    using result_type = bool;
    //!@}

  public:
    // Construct with acceptance probability
    explicit CELER_CONSTEXPR_FUNCTION RejectionSampler(real_type fmax);

    //! Construct when the distribution's maximum is normalized
    CELER_CONSTEXPR_FUNCTION RejectionSampler() : RejectionSampler{1} {}

    // Reject with probability (f/fmax)
    template<class Generator>
    inline CELER_FUNCTION bool operator()(real_type f, Generator& rng) const;

    //! Allow a priori estimate of fmax to be *slightly* unconservative
    static CELER_CONSTEXPR_FUNCTION real_type tolerance()
    {
        return 10 * NumericLimits<real_type>::epsilon();
    }

  private:
    RealType fmax_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with acceptance probability and maximum probability.
 */
template<class RealType>
CELER_CONSTEXPR_FUNCTION
RejectionSampler<RealType>::RejectionSampler(real_type fmax)
    : fmax_{fmax}
{
    CELER_EXPECT(fmax >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Reject with a given probability.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
RejectionSampler<RealType>::operator()(real_type f, Generator& rng) const
    -> result_type
{
    CELER_EXPECT(f <= fmax_ * (1 + tolerance()));
    return f < fmax_ * generate_canonical<RealType>(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
