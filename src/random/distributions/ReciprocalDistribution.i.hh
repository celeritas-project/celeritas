//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ReciprocalDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
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
 * As with UniformRealDistribution, it is allowable for the two bounds to be
 * out of order.
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
ReciprocalDistribution<RealType>::operator()(Generator& rng) -> result_type
{
    return a_ * std::exp(logratio_ * generate_canonical<RealType>(rng));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
