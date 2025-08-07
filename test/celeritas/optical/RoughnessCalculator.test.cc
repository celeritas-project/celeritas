//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/RoughnessCalculator.test.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <random>

#include "corecel/math/ArrayOperators.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/optical/surface/GaussianRoughnessCalculator.hh"
#include "celeritas/optical/surface/SmearRoughnessCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

class RoughnessCalculatorTest : public ::celeritas::test::Test
{
  public:
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    // Sample a roughness calculator and collect distribution moments
    template<class C>
    Array<real_type, 4> sample_moments(C const& sample_normal,
                                       Real3 const& normal,
                                       Real3 const& momentum)
    {
        EXPECT_TRUE(dot_product(normal, momentum) < 0);

        auto& rng = this->rng();

        size_type num_samples = 1000;

        // 1st moment of cos with normal
        // 2nd moment of cos with normal
        // 1st moment of cos with momentum
        // 2nd moment of cos with momentum
        Array<real_type, 4> moments{0, 0, 0, 0};

        for ([[maybe_unused]] auto i : range(num_samples))
        {
            Real3 local_normal = sample_normal(rng);

            real_type normal_cos = dot_product(normal, local_normal);
            real_type momentum_cos = dot_product(momentum, local_normal);

            EXPECT_TRUE(is_soft_unit_vector(local_normal));
            EXPECT_TRUE(momentum_cos < 0);

            moments[0] += normal_cos;
            moments[1] += ipow<2>(normal_cos);
            moments[2] += momentum_cos;
            moments[3] += ipow<2>(momentum_cos);
        }

        moments /= real_type(num_samples);

        return moments;
    }

  private:
    RandomEngine rng_;
};

//---------------------------------------------------------------------------//
// Test smear roughness model distribution
TEST_F(RoughnessCalculatorTest, smear)
{
    real_type roughness = 0.3;
    Real3 normal = make_unit_vector(Real3{0, 0, 1});
    Real3 momentum = make_unit_vector(Real3{0, 1, -1});

    SmearRoughnessCalculator sample_normal(roughness, normal, momentum);

    auto moments = this->sample_moments(sample_normal, normal, momentum);

    Array<real_type, 4> expected_moments{
        0.98210059816854511,
        0.96466393417949625,
        -0.69142901275056512,
        0.48670323495480255,
    };

    EXPECT_VEC_SOFT_EQ(expected_moments, moments);
}

//---------------------------------------------------------------------------//
// Test Gaussian roughness model distribution
TEST_F(RoughnessCalculatorTest, gaussian)
{
    real_type sigma_alpha = 0.3;
    Real3 normal = make_unit_vector(Real3{0, 0, 1});
    Real3 momentum = make_unit_vector(Real3{0, 1, -1});

    GaussianRoughnessCalculator sample_normal(sigma_alpha, normal, momentum);

    auto moments = this->sample_moments(sample_normal, normal, momentum);

    Array<real_type, 4> expected_moments{
        0.955971673785749,
        0.917180750525598,
        -0.670429633303795,
        0.471359786966057,
    };

    EXPECT_VEC_SOFT_EQ(expected_moments, moments);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
