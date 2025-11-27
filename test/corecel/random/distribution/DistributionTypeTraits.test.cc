//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DistributionTypeTraits.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/DistributionTypeTraits.hh"

#include <random>
#include <variant>
#include <vector>

#include "corecel/cont/VariantUtils.hh"
#include "corecel/inp/Distributions.hh"
#include "corecel/random/data/DistributionData.hh"
#include "corecel/random/distribution/DeltaDistribution.hh"
#include "corecel/random/distribution/DistributionInserter.hh"
#include "corecel/random/distribution/DistributionVisitor.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "corecel/random/distribution/UniformBoxDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(DistributionTypeTraitsTest, oned_visit)
{
    using OnedDistribution
        = EnumVariant<OnedDistributionType, OnedDistributionTypeTraits>;

    std::vector<OnedDistribution> distributions;
    distributions.push_back(NormalDistribution<>{1, 0.5});
    distributions.push_back(DeltaDistribution<real_type>{1.23});
    distributions.push_back(DeltaDistribution<real_type>{4.56});

    std::mt19937 rng;
    int num_samples = 4;

    std::vector<real_type> result;
    for (auto& var : distributions)
    {
        for (int i = 0; i < num_samples; ++i)
        {
            result.push_back(
                std::visit([&rng](auto& d) { return d(rng); }, var));
        }
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static double const expected_result[] = {
            1.2258231678182,
            1.1978901974816,
            0.83114664780031,
            1.8521782793476,
            1.23,
            1.23,
            1.23,
            1.23,
            4.56,
            4.56,
            4.56,
            4.56,
        };
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

TEST(DistributionTypeTraitsTest, oned_params)
{
    using VariantDistribution
        = std::variant<inp::DeltaDistribution<double>, inp::NormalDistribution>;

    // Create some distribution inputs
    std::vector<VariantDistribution> distributions;
    distributions.push_back(inp::DeltaDistribution<double>{1.23});
    distributions.push_back(inp::NormalDistribution{10, 1});

    // Construct the distribution params
    HostVal<DistributionParamsData> host;
    DistributionInserter insert(host);
    std::vector<OnedDistributionId> ids;
    for (auto const& var : distributions)
    {
        ids.push_back(std::visit(insert, var));
    }
    HostCRef<DistributionParamsData> params;
    params = host;

    // Sample from the distributions
    DistributionVisitor visit{params};

    std::mt19937 rng;
    int num_samples = 4;

    std::vector<real_type> result;
    for (auto id : ids)
    {
        for (int i = 0; i < num_samples; ++i)
        {
            result.push_back(visit([&rng](auto&& d) { return d(rng); }, id));
        }
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static double const expected_result[] = {
            1.23,
            1.23,
            1.23,
            1.23,
            10.451646335636,
            9.6622932956006,
            11.025567998918,
            10.110686567755,
        };
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

TEST(DistributionTypeTraitsTest, threed_params)
{
    using VariantDistribution
        = std::variant<inp::DeltaDistribution<Array<double, 3>>,
                       inp::IsotropicDistribution,
                       inp::UniformBoxDistribution>;

    // Create some distribution inputs
    std::vector<VariantDistribution> distributions;
    distributions.push_back(
        inp::DeltaDistribution<Array<double, 3>>{{1, 2, 3}});
    distributions.push_back(inp::IsotropicDistribution{});
    distributions.push_back(inp::UniformBoxDistribution{{0, 0, 0}, {1, 1, 1}});

    // Construct the distribution params
    HostVal<DistributionParamsData> host;
    DistributionInserter insert(host);
    std::vector<ThreedDistributionId> ids;
    for (auto const& var : distributions)
    {
        ids.push_back(std::visit(insert, var));
    }
    HostCRef<DistributionParamsData> params;
    params = host;

    // Sample from the distributions
    DistributionVisitor visit{params};

    std::mt19937 rng;
    int num_samples = 4;

    std::vector<Array<real_type, 3>> result;
    for (auto id : ids)
    {
        for (int i = 0; i < num_samples; ++i)
        {
            result.push_back(visit([&rng](auto&& d) { return d(rng); }, id));
        }
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static Array<double, 3> const expected_result[] = {
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {0.34845268346628, -0.58912873788278, -0.72904599140644},
            {0.062868760835021, 0.34161319019477, 0.93773554224846},
            {-0.88312339429508, -0.26998805233565, -0.38366589898599},
            {0.78125163804615, -0.034967222382795, -0.62323604790564},
            {0.99646132554801, 0.9676949370105, 0.72583896321189},
            {0.98110969177694, 0.10986175084421, 0.79810585674955},
            {0.29702944955795, 0.0047834844193157, 0.11246451605618},
            {0.63976335709815, 0.87843064539884, 0.50366267770517},
        };
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
