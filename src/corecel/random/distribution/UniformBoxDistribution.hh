//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/UniformBoxDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/random/data/DistributionData.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample a point uniformly in a box.
 */
template<class RealType = ::celeritas::real_type>
class UniformBoxDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = Array<real_type, 3>;
    using RecordT = UniformBoxDistributionRecord;
    //!@}

  public:
    // Constructor
    inline CELER_FUNCTION
    UniformBoxDistribution(result_type lower, result_type upper);

    // Construct with record
    explicit CELER_FUNCTION UniformBoxDistribution(RecordT const& record)
        : UniformBoxDistribution(record.lower, record.upper)
    {
    }

    // Sample a random point in the box
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    using UniformRealDist = UniformRealDistribution<real_type>;

    Array<UniformRealDist, 3> sample_pos_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from upper and lower coordinates.
 */
template<class RealType>
CELER_FUNCTION
UniformBoxDistribution<RealType>::UniformBoxDistribution(result_type lower,
                                                         result_type upper)
    : sample_pos_{UniformRealDist{lower[0], upper[0]},
                  UniformRealDist{lower[1], upper[1]},
                  UniformRealDist{lower[2], upper[2]}}
{
    CELER_EXPECT(lower[0] <= upper[0]);
    CELER_EXPECT(lower[1] <= upper[1]);
    CELER_EXPECT(lower[2] <= upper[2]);
}

//---------------------------------------------------------------------------//
/*!
 * Sample uniformly in the box.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
UniformBoxDistribution<RealType>::operator()(Generator& rng) -> result_type
{
    result_type result;
    for (int i = 0; i < 3; ++i)
    {
        result[i] = sample_pos_[i](rng);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
