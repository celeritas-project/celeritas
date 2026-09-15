//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/TruncatedDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
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
 *
 * \warning Because of the rejection sampling, it is possible to create a
 * distribution that never (or almost never) accepts a value from the
 * underlying distribution. A loop checker is active in debug mode that
 * prevents more than 32 failed samples. Since GPU performance is so sensitive
 * to rejection failures, consider transforming the underlying distribution if
 * this limit is hit. When using this distribution for physics distributions,
 * it is \em strongly recommended to use \c test::HistogramSampler to verify
 * the number of samples being taken.
 */
template<class Distribution>
class TruncatedDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = typename Distribution::result_type;
    using RecordT = TruncatedDistributionRecord<typename Distribution::RecordT>;
    //!@}

  public:
    // Construct with distribution and truncation bounds
    template<class T, class U, class... Args>
    inline CELER_FUNCTION TruncatedDistribution(
        T lower, U upper, Args&&... args);

    //! Construct from a device-friendly variant record
    explicit CELER_FUNCTION TruncatedDistribution(RecordT const& record)
        : TruncatedDistribution(
              record.lower, record.upper, Distribution(record.distribution))
    {
    }

    // Sample a random number according to the truncated distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //! Assert fewer than this number of samples is tried when CELERITAS_DEBUG
    static constexpr inline int max_debug_samples = 32;

  private:
    Distribution sample_;
    result_type lower_;
    result_type upper_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with distribution and truncation bounds.
 */
template<class Distribution>
template<class T, class U, class... Args>
CELER_FUNCTION TruncatedDistribution<Distribution>::TruncatedDistribution(
    T lower, U upper, Args&&... args)
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
CELER_FUNCTION auto TruncatedDistribution<Distribution>::operator()(
    Generator& rng) -> result_type
{
    int num_remaining_samples = max_debug_samples + 1;
    result_type result;
    do
    {
        if constexpr (CELERITAS_DEBUG)
        {
            // Prevent infinite loops (debug assertions only)
            if (CELER_UNLIKELY(--num_remaining_samples == 0))
            {
                CELER_DEBUG_FAIL(
                    "too many samples taken in TruncatedDistribution",
                    internal);
            }
        }
        else
        {
            // CUDA 12.6 causes warning about unused variable
            CELER_DISCARD(num_remaining_samples);
        }

        result = sample_(rng);
    } while (result < lower_ || result > upper_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
