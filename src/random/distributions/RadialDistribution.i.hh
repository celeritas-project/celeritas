//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
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
