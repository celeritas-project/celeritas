//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/InteractionUtils.hh"

#include <cmath>
#include <random>
#include <vector>

#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(InteractionUtilsTest, calc_exiting_direction)
{
    Real3 inc_dir = {1, 0, 0};
    Real3 out_dir = {0, 1, 0};

    EXPECT_VEC_SOFT_EQ(Real3({real_type(1.0 / std::sqrt(2)),
                              real_type(-1.0 / std::sqrt(2)),
                              0}),
                       calc_exiting_direction({10, inc_dir}, {10, out_dir}));

    EXPECT_VEC_SOFT_EQ(Real3({real_type(1.0 / std::sqrt(101)),
                              real_type(-10.0 / std::sqrt(101)),
                              0}),
                       calc_exiting_direction({1, inc_dir}, {10, out_dir}));
}

TEST(InteractionUtilsTest, sample_exiting_direction)
{
    std::mt19937 rng;
    Real3 inc_dir = make_unit_vector(Real3{1, 2, 3});

    std::vector<real_type> out_dir;
    for (real_type costheta : {-1.0, 0.9, 0.1, 0.0, 0.1, 0.9, 1.0})
    {
        Real3 result = ExitingDirectionSampler{costheta, inc_dir}(rng);
        EXPECT_SOFT_EQ(costheta, dot_product(result, inc_dir));
        out_dir.insert(out_dir.end(), result.begin(), result.end());
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static double const expected_out_dir[]
            = {-0.26726124191242, -0.53452248382485, -0.80178372573727,
               0.65567203926594,  0.47242330622799,  0.58899099879154,
               0.54966700236953,  0.66690027070903,  -0.5031006017034,
               -0.81475551489392, 0.56962591956018,  -0.10816544140881,
               -0.93194807728444, 0.21402106250404,  0.29269056365125,
               0.20505130592048,  0.12514344585105,  0.97071781682466,
               0.26726124191242,  0.53452248382485,  0.80178372573727};
        EXPECT_VEC_SOFT_EQ(expected_out_dir, out_dir);
    }
}

TEST(InteractionUtilsTest, sample_polarization)
{
    DiagnosticRngEngine<std::mt19937> rng;

    std::vector<Real3> direction = {
        {1, 0, 0},
        {0, 0, 1},
        make_unit_vector(Real3{1, 2, 3}),
        make_unit_vector(Real3{1e-3, -1e-3, 1}),
    };
    std::vector<Real3> result;
    for (auto const& dir : direction)
    {
        result.push_back(TransversePolarizationSampler{dir}(rng));
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static Real3 const expected_result[] = {
            {0, -0.6285203911562, -0.77779310738837},
            {0.18099540215833, 0.98348394211474, 0},
            {-0.96349621318245, 0.13467426584993, 0.2313825604942},
            {0.99896510069951, -0.045471273756906, -0.0010444363744566},
        };
        EXPECT_VEC_SOFT_EQ(expected_result, result);
        EXPECT_EQ(4, rng.exchange_count() / real_type(direction.size()));
    }

    size_type num_samples = 10000;
    for (size_type i = 0; i < num_samples; ++i)
    {
        auto dir = IsotropicDistribution{}(rng);
        auto pol = TransversePolarizationSampler{dir}(rng);
        EXPECT_TRUE(is_soft_orthogonal(dir, pol));
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(8, rng.exchange_count() / real_type(num_samples));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
