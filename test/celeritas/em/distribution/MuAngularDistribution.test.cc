//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuAngularDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/MuAngularDistribution.hh"

#include <random>

#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/SampleStats.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(MuAngularDistributionTest, costheta_dist)
{
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;

    Mass muon_mass{105.6583745};
    int num_samples = 1000;

    DiagnosticRngEngine<std::mt19937> rng;
    std::vector<real_type> costheta;
    std::vector<SampleStats> all_stats;

    for (real_type inc_e : {0.1, 1.0, 1e2, 1e3, 1e6})
    {
        for (real_type eps : {0.001, 0.01, 0.1})
        {
            auto stats = sample_distribution(
                MuAngularDistribution(
                    Energy{inc_e}, muon_mass, Energy{eps * inc_e}),
                rng,
                num_samples);
            // stats.print_expected();
            costheta.push_back(stats.mean());
            EXPECT_EQ(2000, rng.exchange_count());
        }
    }

    static double const expected_costheta[] = {
        0.66083519018027,
        0.66173952811755,
        0.65245695633531,
        0.6719537125096,
        0.67562342951953,
        0.65757823745541,
        0.8155925513534,
        0.81631291622027,
        0.80747298359967,
        0.98194125606697,
        0.98314705599445,
        0.98330789156202,
        0.99999997035178,
        0.99999996895068,
        0.99999997195818,
    };
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
