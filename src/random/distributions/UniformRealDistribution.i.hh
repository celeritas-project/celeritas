//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformRealDistribution.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<typename T>
CELER_FUNCTION
UniformRealDistribution<T>::UniformRealDistribution(real_type a, real_type b)
    : a_(a), delta_(b - a)
{
    REQUIRE(a < b);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Sample a random number according to the distribution
 */
template<class T>
template<class Generator>
CELER_FUNCTION auto UniformRealDistribution<T>::operator()(Generator& rng)
    -> result_type
{
    GenerateCanonical<Generator, T> sample_uniform;
    return delta_ * sample_uniform(rng) + a_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
