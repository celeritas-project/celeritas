//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/TsaiUrbanDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"

#include "corecel/random/HistogramSampler.hh"
#include "celeritas/Quantities.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(TsaiUrbanDistributionTest, bin)
{
    using namespace units;

    MevMass const electron_mass = MevMass{0.5109989461};
    constexpr size_type num_samples{10000};

    std::vector<SampledHistogram> actual;

    // Bin cos theta
    HistogramSampler calc_histogram(8, {-1, 1}, num_samples);

    for (real_type inc_e : {0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0})
    {
        TsaiUrbanDistribution sample_mu{MevEnergy{inc_e}, electron_mass};
        actual.push_back(calc_histogram(sample_mu));
    }

    // PRINT_EXPECTED(actual);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        SampledHistogram const expected[] = {
            {{0.1228, 0.1612, 0.1944, 0.2712, 0.3432, 0.4772, 0.8148, 1.6152},
             7.9242},
            {{0.1024, 0.128, 0.1744, 0.2012, 0.296, 0.4524, 0.7956, 1.85},
             7.413},
            {{0.0504, 0.0568, 0.0696, 0.1032, 0.1592, 0.2816, 0.636, 2.6432},
             6.552},
            {{0.0292, 0.0292, 0.0428, 0.0648, 0.088, 0.1616, 0.3788, 3.2056},
             6.2022},
            {{0, 0, 0, 0.0008, 0.0044, 0.006, 0.034, 3.9548}, 6},
            {{0, 0, 0, 0, 0, 0, 0.0004, 3.9996}, 6},
            {{0, 0, 0, 0, 0, 0, 0, 4}, 6},
        };
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
