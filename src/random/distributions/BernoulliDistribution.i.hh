//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BernoulliDistribution.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the probability of returning true.
 */
CELER_FUNCTION BernoulliDistribution::BernoulliDistribution(real_type p_true)
    : p_true_(p_true)
{
    CELER_EXPECT(p_true >= 0 && p_true <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the UNnormalized probability of returning true or false
 */
CELER_FUNCTION
BernoulliDistribution::BernoulliDistribution(real_type scaled_true,
                                             real_type scaled_false)
    : p_true_(scaled_true / (scaled_true + scaled_false))
{
    CELER_EXPECT(scaled_true > 0 || scaled_false > 0);
    CELER_EXPECT(scaled_true >= 0 && scaled_false >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the probability of returning true.
 */
template<class Generator>
CELER_FUNCTION auto BernoulliDistribution::operator()(Generator& rng)
    -> result_type
{
    return generate_canonical<real_type>(rng) < p_true_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
