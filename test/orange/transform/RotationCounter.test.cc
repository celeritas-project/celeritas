//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/RotationCounter.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/RotationCounter.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/Turn.hh"
#include "orange/MatrixUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(RotationCounterTest, unrotated)
{
    RotationCounter rc;
    rc(NoTransformation{});
    rc(Translation{Real3{1, 2, 3}});
    rc(Transformation{make_identity(), Real3{0, 0, 1}});
    EXPECT_EQ(1, rc.num_rotations());

    auto result = rc.calc_most_common();
    EXPECT_EQ(make_identity(), result.rotation);
    EXPECT_EQ(3, result.count);
}

TEST(RotationCounterTest, shared_rotations)
{
    auto const rot_0 = make_rotation(Axis::x, Turn{0.1});
    auto const rot_1 = make_rotation(Axis::x, Turn{0.2});

    RotationCounter rc;

    // Insert 10 of rot_0 and 9 of rot_1
    for (auto i : range(10))
    {
        rc(Transformation{rot_0, Real3{0, 0, static_cast<real_type>(i)}});
    }
    for (auto i : range(9))
    {
        rc(Transformation{rot_1, Real3{0, 0, static_cast<real_type>(i)}});
    }

    EXPECT_EQ(2, rc.num_rotations());

    // Check that we get 10 rot_0s
    auto result = rc.calc_most_common();
    EXPECT_EQ(rot_0, result.rotation);
    EXPECT_EQ(10, result.count);
}

TEST(RotationCounterTest, distinct)
{
    RotationCounter rc;
    for (auto i : range(8))
    {
        Turn const turn{static_cast<real_type>(0.05 * i)};
        rc(Transformation{make_rotation(Axis::x, turn), Real3{0, 0, 0}});
    }
    EXPECT_EQ(8, rc.num_rotations());

    // No rotation is shared
    EXPECT_EQ(1, rc.calc_most_common().count);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
