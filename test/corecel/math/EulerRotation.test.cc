//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/EulerRotation.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/EulerRotation.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class EulerRotationTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
TEST_F(EulerRotationTest, rotate_pi)
{
    Real3 init_dirx{1, 0, 0};
    EulerRotation rotate_x(constants::pi, 0, 0);
    auto result_x = rotate_x(init_dirx);

    static Real3 const expected_result_x{-1, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_result_x, result_x);
}

TEST_F(EulerRotationTest, rotate_full)
{
    auto const twopi = 2 * constants::pi;

    Real3 init_dir{1, 1, 1};
    EulerRotation rotate_full(twopi, twopi, twopi);
    auto result_unchanged = rotate_full(init_dir);

    EXPECT_VEC_SOFT_EQ(result_unchanged, init_dir);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
