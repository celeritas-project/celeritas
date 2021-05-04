//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformRealDistribution.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [0, 1).
 */
template<class RealType>
CELER_FUNCTION UniformRealDistribution<RealType>::UniformRealDistribution()
    : UniformRealDistribution(0)
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
