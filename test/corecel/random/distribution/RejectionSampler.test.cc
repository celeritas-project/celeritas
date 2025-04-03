//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/RejectionSampler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/RejectionSampler.hh"

#include <random>

#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Target PDF with domain [0, 2] and range [0.5, 2]
double target_distribution(double x)
{
    CELER_EXPECT(x >= 0 && x <= 2);
    if (x < 0.5)
        return 1.0;
    if (x < 1)
        return 2.0;
    return 0.5;
}

struct TargetSampler
{
    UniformRealDistribution<double> sample_domain{0, 2};

    template<class Engine>
    real_type operator()(Engine& rng)
    {
        real_type x;
        do
        {
            x = this->sample_domain(rng);
        } while (RejectionSampler<double>{target_distribution(x), 2.0}(rng));
        return x;
    }
};

TEST(RejectionSamplerTest, sample)
{
    DiagnosticRngEngine<std::mt19937> rng;
    constexpr int num_samples = 16000;

    TargetSampler sample_target;

    std::vector<int> counters(4);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double x = sample_target(rng);
        ASSERT_GE(x, 0);
        ASSERT_LT(x, 2);
        auto idx = static_cast<std::size_t>((x / 2.0) * counters.size());
        counters.at(idx) += 1;
    }

    int const expected_counters[] = {3942, 7996, 2034, 2028};
    EXPECT_VEC_EQ(expected_counters, counters);

    EXPECT_EQ(127408, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
