//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/UniformRealDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a uniform distribution.
 *
 * This distribution is defined between two arbitrary real numbers \em a and
 * \em b , and has a flat PDF between the two values. It \em is allowable for
 * the two numbers to have reversed order.
 * The normalized PDF is:
 * \f[
   f(x; a, b) = \frac{1}{b - a} \quad \mathrm{for} \ a \le x < b
   \f]
 * which integrated into a CDF and inverted gives a sample:
 * \f[
  x = (b - a) \xi + a
   \f]
 */
template<class RealType = ::celeritas::real_type>
class UniformRealDistribution
{
    static_assert(std::is_floating_point_v<RealType>);

  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on [0, 1)
    CELER_CONSTEXPR_FUNCTION UniformRealDistribution();

    // Construct on an arbitrary interval
    CELER_CONSTEXPR_FUNCTION UniformRealDistribution(real_type a, real_type b);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng) const;

    //// ACCESSORS ////

    //! Get the lower bound of the distribution
    CELER_CONSTEXPR_FUNCTION real_type a() const { return a_; }

    //! Get the upper bound of the distribution
    CELER_CONSTEXPR_FUNCTION real_type b() const { return delta_ + a_; }

  private:
    RealType a_;
    RealType delta_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [0, 1).
 *
 * This constructor is generally unused because it's simpler and more efficient
 * to directly call ``generate_canonical``. We leave it for compatibility with
 * the standard.
 */
template<class RealType>
CELER_CONSTEXPR_FUNCTION
UniformRealDistribution<RealType>::UniformRealDistribution()
    : UniformRealDistribution(0, 1)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [a, b).
 *
 * Note that it's allowable for these two to be out of order to support other
 * generators (inverse square, power) where they may be inverted and out of
 * order.
 */
template<class RealType>
CELER_CONSTEXPR_FUNCTION
UniformRealDistribution<RealType>::UniformRealDistribution(real_type a,
                                                           real_type b)
    : a_(a), delta_(b - a)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
UniformRealDistribution<RealType>::operator()(Generator& rng) const -> result_type
{
    return std::fma(delta_, generate_canonical<RealType>(rng), a_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
