//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/TsaiUrbanDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"

#include <random>

#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(TsaiUrbanDistributionTest, bin)
{
    using namespace constants;
    using namespace units;

    MevMass const electron_mass = MevMass{0.5109989461};
    constexpr size_type num_samples{10000};

    std::vector<std::vector<double>> angle_dist;
    std::vector<double> avg_rng_count;

    // Loop over various electron energies
    DiagnosticRngEngine<std::mt19937> rng;
    for (real_type inc_e : {0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0})
    {
        Histogram bin_angle(8, {-1, 1});
        accumulate_n(bin_angle,
                     TsaiUrbanDistribution{MevEnergy{inc_e}, electron_mass},
                     rng,
                     num_samples);

        EXPECT_EQ(0, bin_angle.underflow())
            << "Encountered values as low as " << bin_angle.min();
        EXPECT_EQ(0, bin_angle.overflow())
            << "Encountered values as high as " << bin_angle.max();

        angle_dist.push_back(bin_angle.calc_density());
        avg_rng_count.push_back(rng.exchange_count()
                                / static_cast<double>(num_samples));
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static std::vector<double> const expected_angle_dist[] = {
            {0.1228, 0.1612, 0.1944, 0.2712, 0.3432, 0.4772, 0.8148, 1.6152},
            {0.1024, 0.128, 0.1744, 0.2012, 0.296, 0.4524, 0.7956, 1.85},
            {0.0504, 0.0568, 0.0696, 0.1032, 0.1592, 0.2816, 0.636, 2.6432},
            {0.0292, 0.0292, 0.0428, 0.0648, 0.088, 0.1616, 0.3788, 3.2056},
            {0, 0, 0, 0.0008, 0.0044, 0.006, 0.034, 3.9548},
            {0, 0, 0, 0, 0, 0, 0.0004, 3.9996},
            {0, 0, 0, 0, 0, 0, 0, 4}};
        EXPECT_VEC_SOFT_EQ(expected_angle_dist, angle_dist);
        static double const expected_avg_rng_count[]
            = {7.9242, 7.413, 6.552, 6.2022, 6, 6, 6};
        EXPECT_VEC_SOFT_EQ(expected_avg_rng_count, avg_rng_count);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
