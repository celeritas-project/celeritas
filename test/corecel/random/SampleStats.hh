//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/SampleStats.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <array>
#include <iosfwd>
#include <limits>
#include <gtest/gtest.h>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Accumulate sample statistics.
 */
class SampleStatsAccumulator
{
  public:
    using real_type = double;

    static constexpr size_type num_moments = 3;

    // Default constructor
    SampleStatsAccumulator() = default;

    // Accumulate a value
    inline void operator()(real_type value);

    //!@{
    // Access accumulated values
    real_type min() const { return min_; }
    real_type max() const { return max_; }
    size_type count() const { return count_; }
    inline real_type moment(size_type i) const;
    //!@}

  private:
    real_type min_{std::numeric_limits<real_type>::infinity()};
    real_type max_{-std::numeric_limits<real_type>::infinity()};
    size_type count_{0};  // zeroth moment
    std::array<real_type, num_moments - 1> moments_{};
};

//---------------------------------------------------------------------------//
//! Accumulate a single value
void SampleStatsAccumulator::operator()(real_type value)
{
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
    real_type value_pow = value;
    for (size_type i = 0; i < num_moments - 1; ++i)
    {
        moments_[i] += value_pow;
        value_pow *= value;
    }
    ++count_;
}

//---------------------------------------------------------------------------//
//! Get the ith moment
auto SampleStatsAccumulator::moment(size_type i) const -> real_type
{
    if (i == 0)
        return static_cast<real_type>(count_);
    return moments_[i - 1];
}

//---------------------------------------------------------------------------//
/*!
 * Final computed statistics from accumulator.
 */
class SampleStats
{
  public:
    using real_type = SampleStatsAccumulator::real_type;

    // Construct from accumulator
    explicit SampleStats(SampleStatsAccumulator acc);

    // Construct from calculated values
    SampleStats(real_type min,
                real_type max,
                real_type mean,
                real_type stdev,
                size_type count);

    real_type min() const { return min_; }
    real_type max() const { return max_; }
    real_type mean() const { return mean_; }
    real_type stdev() const { return stdev_; }
    size_type count() const { return count_; }

    // Calculate the difference, in stdev, between this and another
    // distribution
    real_type calc_z_score(SampleStats const& other) const;

    // Calculate F-statistic for variance comparison
    real_type calc_f_statistic(SampleStats const& other) const;

    // Determine if two sample statistics are statistically equivalent
    bool probably_equal(SampleStats const& other,
                        real_type confidence = 0.95) const;

    // Test if variances are probably equal
    bool probably_equal_variance(SampleStats const& other,
                                 real_type confidence) const;

    void print_expected() const;

  private:
    real_type min_;
    real_type max_;
    real_type mean_;
    real_type stdev_;
    size_type count_;
};

//---------------------------------------------------------------------------//
/*!
 * Sample statistics from a random distribution and collect statistics.
 */
template<typename DistributionT, typename RandomEngineT>
SampleStats sample_distribution(DistributionT&& sample_from,
                                RandomEngineT& engine,
                                size_type count)
{
    SampleStatsAccumulator accumulate;
    for (size_type i = 0; i < count; ++i)
    {
        accumulate(sample_from(engine));
    }
    return SampleStats(std::move(accumulate));
}

//---------------------------------------------------------------------------//
// Print to a stream
std::ostream& operator<<(std::ostream&, SampleStats const& ss);

//---------------------------------------------------------------------------//
struct SampleStatsTolerance
{
    //! Confidence level for mean
    real_type mean_confidence = 0.95;
    //! "Soft equal" tolerance for variance
    real_type stdev_tolerance = 0.1;
};

//---------------------------------------------------------------------------//
// Test functions
::testing::AssertionResult IsRefEqual(char const* expected_expr,
                                      char const* actual_expr,
                                      char const* conf_expr,
                                      SampleStats const& expected,
                                      SampleStats const& actual,
                                      SampleStatsTolerance tol);

inline ::testing::AssertionResult IsRefEqual(char const* expected_expr,
                                             char const* actual_expr,
                                             SampleStats const& expected,
                                             SampleStats const& actual)
{
    return IsRefEqual(
        expected_expr, actual_expr, "default", expected, actual, {});
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
