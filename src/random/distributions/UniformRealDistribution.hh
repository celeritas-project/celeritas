//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformRealDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a uniform distribution.
 *
 * This distribution is defined between two arbitrary real numbers \em a and
 * \em b , and has a flat PDF between the two values. It *is* allowable for the
 * two numbers to have reversed order.
 */
template<class RealType = ::celeritas::real_type>
class UniformRealDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on [0, 1)
    inline CELER_FUNCTION UniformRealDistribution();

    // Construct on an arbitrary interval
    inline CELER_FUNCTION UniformRealDistribution(real_type a, real_type b);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //// ACCESSORS ////

    //! Get the lower bound of the distribution
    CELER_FUNCTION real_type a() const { return a_; }

    //! Get the upper bound of the distribution
    CELER_FUNCTION real_type b() const { return delta_ + a_; }

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
CELER_FUNCTION UniformRealDistribution<RealType>::UniformRealDistribution()
    : UniformRealDistribution(0, 1)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [a, b).
 */
template<class RealType>
CELER_FUNCTION
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
UniformRealDistribution<RealType>::operator()(Generator& rng) -> result_type
{
    return delta_ * generate_canonical<RealType>(rng) + a_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
