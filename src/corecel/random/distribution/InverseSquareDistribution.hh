//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/InverseSquareDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample \f$ 1/x^2 \f$ over a given domain.
 *
 * This distribution is defined on a positive range \f$ [a, b) \f$ and has the
 * normalized PDF:
 * \f[
   f(x; a, b) = \frac{a b}{x^2 (b - a)} \quad \mathrm{for} \  a \le x < b
   \f]
 * which integrated into a CDF and inverted gives a sample:
 * \f[
  x = \frac{a b}{a + \xi (b - a)}
   \f]
 */
template<class RealType = ::celeritas::real_type>
class InverseSquareDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on an interval
    inline CELER_FUNCTION InverseSquareDistribution(real_type a, real_type b);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng) const;

  private:
    RealType product_;
    UniformRealDistribution<RealType> sample_denom_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [a, b).
 *
 * As with UniformRealDistribution, it is allowable for the two bounds to be
 * out of order.
 */
template<class RealType>
CELER_FUNCTION
InverseSquareDistribution<RealType>::InverseSquareDistribution(real_type a,
                                                               real_type b)
    : product_{a * b}, sample_denom_{a, b}
{
    CELER_EXPECT(a > 0);
    CELER_EXPECT(b >= a);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
InverseSquareDistribution<RealType>::operator()(Generator& rng) const -> result_type
{
    return product_ / sample_denom_(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
