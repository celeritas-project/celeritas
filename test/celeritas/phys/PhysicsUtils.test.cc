//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/PhysicsUtils.hh"

#include <cmath>
#include <random>
#include <vector>

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(PhysicsUtilsTest, calc_exiting_direction)
{
    Real3 inc_dir = {1, 0, 0};
    Real3 out_dir = {0, 1, 0};

    Real3 result = calc_exiting_direction({10, inc_dir}, {10, out_dir});
    Real3 expected = {1.0 / std::sqrt(2), -1.0 / std::sqrt(2), 0};
    EXPECT_VEC_SOFT_EQ(expected, result);

    result = calc_exiting_direction({1, inc_dir}, {10, out_dir});
    expected = {1.0 / std::sqrt(101), -10.0 / std::sqrt(101), 0};
    EXPECT_VEC_SOFT_EQ(expected, result);
}

TEST(PhysicsUtilsTest, sample_exiting_direction)
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
