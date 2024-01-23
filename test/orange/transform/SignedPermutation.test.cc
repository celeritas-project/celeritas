//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/SignedPermutation.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/SignedPermutation.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
std::string to_string(SignedPermutation::SignedAxes const& p)
{
    std::string result;
    for (auto ax : range(Axis::size_))
    {
        result.push_back(p[ax].first);
        result.push_back(to_char(p[ax].second));
        result.push_back(',');
    }
    result.pop_back();
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Test harness for signed permutation.
 *
 * Note that the rotation matrix is a quarter turn around Z, then a quarter
 * turn around X.
 \code
    auto r = make_rotation(Axis::x, Turn{0.25},
                           make_rotation(Axis::z, Turn{0.25}));
    int expected_r[] = {0, -1, 0,
                        0, 0, -1,
                        1, 0, 0};
 * \endcode
 *
 */
class SignedPermutationTest : public ::celeritas::test::Test
{
  public:
    using SignedAxes = SignedPermutation::SignedAxes;

    SignedPermutation sp_zx{
        SignedAxes{{{'-', Axis::y}, {'-', Axis::z}, {'+', Axis::x}}}};
};

TEST_F(SignedPermutationTest, construction)
{
    {
        SCOPED_TRACE("default construct");
        SignedPermutation sp;
        EXPECT_EQ("+x,+y,+z", to_string(sp.permutation()));
    }
    {
        SCOPED_TRACE("permutation vector");
        EXPECT_EQ("-y,-z,+x", to_string(sp_zx.permutation()));
    }
    {
        SCOPED_TRACE("serialize");
        SignedPermutation sp2{make_span(sp_zx.data())};
        EXPECT_EQ(to_string(sp_zx.permutation()), to_string(sp2.permutation()));
    }
}

TEST_F(SignedPermutationTest, invalid_construction)
{
    {
        SCOPED_TRACE("invalid sign");
        EXPECT_THROW(SignedPermutation(SignedAxes{
                         {{'?', Axis::y}, {'-', Axis::z}, {'+', Axis::x}}}),
                     RuntimeError);
    }
    {
        SCOPED_TRACE("singular matrix");
        EXPECT_THROW(SignedPermutation(SignedAxes{
                         {{'?', Axis::y}, {'-', Axis::z}, {'+', Axis::y}}}),
                     RuntimeError);
    }
    {
        SCOPED_TRACE("reflecting is prohibited");
        EXPECT_THROW(SignedPermutation(SignedAxes{
                         {{'+', Axis::y}, {'-', Axis::z}, {'+', Axis::x}}}),
                     RuntimeError);
    }
}

TEST_F(SignedPermutationTest, transform)
{
    {
        SignedPermutation sp = make_permutation(Axis::z, QuarterTurn{1});
        // Daughter to parent: rotate quarter turn around Z
        EXPECT_VEC_EQ((Real3{-3, 2, 0}), sp.transform_up({2, 3, 0}));
        // Parent to daughter: rotate back
        EXPECT_VEC_EQ((Real3{2, 3, 0}), sp.transform_down({-3, 2, 0}));
    }
    {
        auto const daughter = Real3{1, 2, 3};
        auto const parent = sp_zx.transform_up(daughter);
        EXPECT_VEC_SOFT_EQ((Real3{-2, -3, 1}), parent);
        EXPECT_VEC_SOFT_EQ(daughter, sp_zx.transform_down(parent));
    }
}

TEST_F(SignedPermutationTest, rotate)
{
    {
        // Daughter to parent: rotate quarter turn around Z
        EXPECT_VEC_EQ((Real3{-2, -3, 1}), sp_zx.rotate_up({1, 2, 3}));
        // Parent to daughter: rotate back
        EXPECT_VEC_EQ((Real3{1, 2, 3}), sp_zx.rotate_down({-2, -3, 1}));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
