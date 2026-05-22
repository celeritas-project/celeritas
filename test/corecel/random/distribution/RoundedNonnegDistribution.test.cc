//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/RoundedNonnegDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/RoundedNonnegDistribution.hh"

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

namespace
{
struct ConstantDistribution
{
    using real_type = double;
    using result_type = double;

    double value;

    explicit CELER_FUNCTION ConstantDistribution(double v) : value(v) {}

    template<class Generator>
    CELER_FUNCTION double operator()(Generator&)
    {
        return value;
    }
};

struct CountingDistribution
{
    using real_type = double;
    using result_type = double;

    int* count;
    double value;

    CELER_FUNCTION CountingDistribution(int* c, double v) : count(c), value(v)
    {
    }

    template<class Generator>
    CELER_FUNCTION double operator()(Generator&)
    {
        ++(*count);
        return value;
    }
};
}  // namespace

TEST(RoundedNonnegDistributionTest, rounding)
{
    struct DummyRng
    {
    } rng;

    using RoundedConstant = RoundedNonnegDistribution<ConstantDistribution>;
    static_assert(
        std::is_same_v<RoundedConstant::result_type, celeritas::size_type>);

    EXPECT_EQ(0, RoundedConstant{-4.2}(rng));
    EXPECT_EQ(0, RoundedConstant{-0.2}(rng));
    EXPECT_EQ(0, RoundedConstant{0.49}(rng));
    EXPECT_EQ(1, RoundedConstant{0.5}(rng));
    EXPECT_EQ(2, RoundedConstant{1.5}(rng));
    EXPECT_EQ(3, RoundedConstant{2.5}(rng));

    using RoundedU8
        = RoundedNonnegDistribution<ConstantDistribution, std::uint8_t>;
    EXPECT_EQ(std::numeric_limits<std::uint8_t>::max(), RoundedU8{1e6}(rng));
}

TEST(RoundedNonnegDistributionTest, one_sample_per_call)
{
    struct DummyRng
    {
    } rng;

    int num_samples = 10000;
    int num_distribution_samples = 0;
    RoundedNonnegDistribution<CountingDistribution> sample_counting{
        &num_distribution_samples, 1.25};

    for ([[maybe_unused]] int i : range(num_samples))
    {
        EXPECT_EQ(1, sample_counting(rng));
    }

    EXPECT_EQ(num_samples, num_distribution_samples);
}

TEST(RoundedNonnegDistributionTest, uniform)
{
    using RoundedUniform
        = RoundedNonnegDistribution<UniformRealDistribution<double>>;

    int num_samples = 10000;
    DiagnosticRngEngine<std::mt19937> rng;
    RoundedUniform sample_uniform{-0.8, 4.2};

    Histogram histogram(5, {0, 5});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(static_cast<double>(sample_uniform(rng)));
    }

    static unsigned int const expected_counts[]
        = {2673u, 1898u, 2078u, 1974u, 1377u};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(0, histogram.underflow());
    EXPECT_EQ(0, histogram.overflow());
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
