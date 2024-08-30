//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/NormalDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/NormalDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(NormalDistributionTest, normal)
{
    DiagnosticRngEngine<std::mt19937> rng;
    int num_samples = 10000;

    NormalDistribution<double> sample_normal{/* mean = */ 0.0,
                                             /* stddev = */ 1.0};
    std::vector<int> counters(6);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double x = sample_normal(rng);
        if (x < -2.0)
            ++counters[0];
        else if (x < -1.0)
            ++counters[1];
        else if (x < 0.0)
            ++counters[2];
        else if (x < 1.0)
            ++counters[3];
        else if (x < 2.0)
            ++counters[4];
        else
            ++counters[5];
    }

    int const expected_counters[] = {235, 1379, 3397, 3411, 1352, 226};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(2 * num_samples, rng.count());
}

TEST(NormalDistributionTest, move)
{
    DiagnosticRngEngine<std::mt19937> rng;
    NormalDistribution<double> sample_normal{/* mean = */ 0,
                                             /* stddev = */ 0.5};

    std::vector<double> samples;
    for ([[maybe_unused]] int i : range(4))
    {
        samples.push_back(sample_normal(rng));
    }

    // Check that resetting RNG gives same results
    rng = {};
    for ([[maybe_unused]] int i : range(4))
    {
        EXPECT_DOUBLE_EQ(samples[i], sample_normal(rng));
    }

    // Replace after 1 sample: should be scaled original (using latent spare)
    rng = {};
    EXPECT_DOUBLE_EQ(samples[0], sample_normal(rng));
    sample_normal = {1.0, 1.0};  // Shift right, double width
    EXPECT_DOUBLE_EQ(2 * samples[1] + 1, sample_normal(rng));

    // Check that we capture the "spare" value from another distribution
    sample_normal = [] {
        NormalDistribution<double> sample_other_normal{0, 2.0};
        std::mt19937 temp_rng;
        sample_other_normal(temp_rng);
        return sample_other_normal;
    }();
    EXPECT_DOUBLE_EQ(4 * samples[1], sample_normal(rng));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
