//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NormalDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with mean and standard deviation.
 */
template<class RealType>
CELER_FUNCTION
NormalDistribution<RealType>::NormalDistribution(real_type mean,
                                                 real_type stddev)
    : mean_(mean), stddev_(stddev), spare_(0), has_spare_(false)
{
    CELER_EXPECT(stddev > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto NormalDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    if (has_spare_)
    {
        has_spare_ = false;
        return spare_ * stddev_ + mean_;
    }

    real_type theta = 2 * constants::pi * generate_canonical(rng);
    real_type r     = std::sqrt(-2 * std::log(generate_canonical(rng)));
    spare_          = r * std::cos(theta);
    has_spare_      = true;
    return r * std::sin(theta) * stddev_ + mean_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
