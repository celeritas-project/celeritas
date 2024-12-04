//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuAngularDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/MuAngularDistribution.hh"

#include <random>

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
    std::vector<real_type> costheta;
    std::mt19937 rng;

    for (real_type inc_e : {0.1, 1.0, 1e2, 1e3, 1e6})
    {
        for (real_type eps : {0.001, 0.01, 0.1})
        {
            MuAngularDistribution sample_costheta(
                Energy{inc_e}, muon_mass, Energy{eps * inc_e});

            real_type costheta_sum = 0;
            for (int i = 0; i < num_samples; ++i)
            {
                costheta_sum += sample_costheta(rng);
            }
            costheta.push_back(costheta_sum / num_samples);
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
