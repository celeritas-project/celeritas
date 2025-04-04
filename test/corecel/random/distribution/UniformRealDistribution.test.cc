//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/UniformRealDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/UniformRealDistribution.hh"

#include <random>

#include "celeritas_test.hh"
#include "../DiagnosticRngEngine.hh"
#include "../SampleStats.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(UniformRealDistributionTest, constructors)
{
    {
        UniformRealDistribution<> sample_uniform{};
        EXPECT_SOFT_EQ(0.0, sample_uniform.a());
        EXPECT_SOFT_EQ(1.0, sample_uniform.b());
    }
    {
        UniformRealDistribution<> sample_uniform{1, 2};
        EXPECT_SOFT_EQ(1.0, sample_uniform.a());
        EXPECT_SOFT_EQ(2.0, sample_uniform.b());
    }
    if (CELERITAS_DEBUG)
    {
        // b < a is not allowed
        EXPECT_THROW(UniformRealDistribution<>(3, 2), DebugError);
    }
}

TEST(UniformRealDistributionTest, distribution)
{
    double min = 0.0;
    double max = 5.0;
    unsigned int num_samples = 10000;

    DiagnosticRngEngine<std::mt19937> rng;
    auto stats = sample_distribution(
        UniformRealDistribution<double>{min, max}, rng, num_samples);
    // stats.print_expected();

    // Exact reference stats for a uniform distribution on [0,5]:
    // - Mean should be 2.5
    // - Standard deviation should be 5/sqrt(12) ≈ 1.443
    SampleStats ref(min, max, 2.5, 5.0 / std::sqrt(12.0), 1000000);

    // Compare sample stats to reference
    EXPECT_REF_EQ(ref, stats);
    EXPECT_EQ(20000, rng.exchange_count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
