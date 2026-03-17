//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/NormalDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/NormalDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"

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

    double mean = 0.0;
    double stddev = 1.0;
    NormalDistribution<double> sample_normal{mean, stddev};

    Histogram histogram(8, {-4, 4});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(sample_normal(rng));
    }
    static unsigned int const expected_counts[]
        = {17, 218, 1379, 3397, 3411, 1352, 211, 15};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
