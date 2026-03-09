//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/TruncatedDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/random/data/DistributionData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a truncated distribution.
 *
 * Sample from an arbitrary distribution truncated to a finite interval. Values
 * are drawn from the underlying distribution using rejection sampling until
 * they fall within the truncation bounds.
 */
template<class Distribution>
class TruncatedDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = typename Distribution::real_type;
    using result_type = typename Distribution::result_type;
    using RecordT = TruncatedDistributionRecord<typename Distribution::RecordT>;
    //!@}

  public:
    // Construct with distribution and truncation bounds
    template<class... Args>
    inline CELER_FUNCTION
    TruncatedDistribution(real_type lower, real_type upper, Args&&... args);

    // Construct from record
    explicit CELER_FUNCTION TruncatedDistribution(RecordT const& record)
        : TruncatedDistribution(
              record.lower, record.upper, Distribution(record.distribution))
    {
    }

    // Sample a random number according to the truncated distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    Distribution sample_;
    real_type lower_;
    real_type upper_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with distribution and truncation bounds.
 */
template<class Distribution>
template<class... Args>
CELER_FUNCTION
TruncatedDistribution<Distribution>::TruncatedDistribution(real_type lower,
                                                           real_type upper,
                                                           Args&&... args)
    : sample_(celeritas::forward<Args>(args)...), lower_(lower), upper_(upper)
{
    CELER_EXPECT(lower < upper);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the truncated distribution.
 */
template<class Distribution>
template<class Generator>
CELER_FUNCTION auto
TruncatedDistribution<Distribution>::operator()(Generator& rng) -> result_type
{
    result_type result;
    do
    {
        // Reject samples outside the truncation bounds
        result = sample_(rng);
    } while (result < lower_ || result > upper_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
