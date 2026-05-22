//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/RoundedNonnegDistribution.hh
//! \sa RoundedNonnegDistribution.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a distribution and return a rounded non-negative integer.
 *
 * This distribution wraps an arbitrary underlying sampler and rounds each
 * value to the nearest integer by adding 0.5 and truncating. Negative values
 * are clamped to zero, and values above the integer type range are clamped to
 * the maximum representable value.
 */
template<class Distribution, class IntType = celeritas::size_type>
class RoundedNonnegDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = typename Distribution::real_type;
    using result_type = IntType;
    //!@}

    static_assert(std::is_floating_point_v<real_type>);
    static_assert(std::is_integral_v<result_type>);

  public:
    // Construct with distribution arguments
    template<class... Args>
    inline CELER_FUNCTION explicit RoundedNonnegDistribution(Args&&... args);

    // Sample a rounded non-negative integer
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    Distribution sample_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with arguments forwarded to the wrapped distribution.
 */
template<class Distribution, class IntType>
template<class... Args>
CELER_FUNCTION
RoundedNonnegDistribution<Distribution, IntType>::RoundedNonnegDistribution(
    Args&&... args)
    : sample_(celeritas::forward<Args>(args)...)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the rounded non-negative distribution.
 */
template<class Distribution, class IntType>
template<class Generator>
CELER_FUNCTION auto
RoundedNonnegDistribution<Distribution, IntType>::operator()(Generator& rng)
    -> result_type
{
    real_type value = sample_(rng) + real_type{0.5};
    value = clamp(
        value,
        real_type{0},
        static_cast<real_type>(std::numeric_limits<result_type>::max()));
    return static_cast<result_type>(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
