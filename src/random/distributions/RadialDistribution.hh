//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a uniform radial distribution.
 */
template<class RealType = ::celeritas::real_type>
class RadialDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Constructor
    explicit inline CELER_FUNCTION RadialDistribution(real_type radius);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //// ACCESSORS ////

    //! Get the sampling radius
    inline CELER_FUNCTION real_type radius() const { return radius_; }

  private:
    RealType radius_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<class RealType>
CELER_FUNCTION
RadialDistribution<RealType>::RadialDistribution(real_type radius)
    : radius_(radius)
{
    CELER_EXPECT(radius_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto RadialDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    return std::cbrt(generate_canonical<RealType>(rng)) * radius_;
}
//---------------------------------------------------------------------------//
} // namespace celeritas
