//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/ReciprocalDistribution.hh
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
 * Reciprocal or log-uniform distribution.
 *
 * This distribution is defined on a positive range \f$ [a, b) \f$ and has the
 * normalized PDF:
 * \f[
   f(x; a, b) = \frac{1}{x (\ln b - \ln a)} \quad \mathrm{for} \ a \le x < b
   \f]
 * which integrated into a CDF and inverted gives a sample:
 * \f[
  x = a \left( \frac{b}{a} \right)^{\xi}
    = a \exp\!\left(\xi \log \frac{b}{a} \right)
   \f]
 */
template<class RealType = ::celeritas::real_type>
class ReciprocalDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on an the interval [a, 1]
    explicit inline CELER_FUNCTION ReciprocalDistribution(real_type a);

    // Construct on an arbitrary interval
    inline CELER_FUNCTION ReciprocalDistribution(real_type a, real_type b);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng) const;

  private:
    RealType a_;
    RealType logratio_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [a, 1).
 *
 * The distribution is equivalent to switching a and b, and using
 * \f$ \xi' = 1 - \xi \f$.
 */
template<class RealType>
CELER_FUNCTION
ReciprocalDistribution<RealType>::ReciprocalDistribution(real_type a)
    : ReciprocalDistribution(1, a)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [a, b).
 *
 * It is allowable for the two bounds to be out of order.
 *
 * Note that writing as \code (1/a) * b \endcode allows the compiler to
 * optimize better for the constexpr case a=1.
 */
template<class RealType>
CELER_FUNCTION
ReciprocalDistribution<RealType>::ReciprocalDistribution(real_type a,
                                                         real_type b)
    : a_(a), logratio_(std::log((1 / a) * b))
{
    CELER_EXPECT(a > 0);
    CELER_EXPECT(b > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
ReciprocalDistribution<RealType>::operator()(Generator& rng) const
    -> result_type
{
    return a_ * std::exp(logratio_ * generate_canonical<RealType>(rng));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
