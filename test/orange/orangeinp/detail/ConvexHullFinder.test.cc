//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ConvexHullFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/ConvexHullFinder.hh"

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
/*
 * Test harness for ConvexHullFinder
 */
class ConvexHullFinderTest : public ::celeritas::test::Test
{
  public:
    using CHF = ConvexHullFinder<double>;
    using VecReal2 = CHF::VecReal2;
    using VecVecReal2 = CHF::VecVecReal2;

    ConvexHullFinderTest()
    {
        t.rel = 1e-6;
        t.abs = 1e-10;
    }

    void expect_eq(VecReal2 const& expected, VecReal2 const& actual)
    {
        ASSERT_EQ(expected.size(), actual.size());
        for (auto i : range(expected.size()))
        {
            EXPECT_EQ(expected[i][0], actual[i][0]);
            EXPECT_EQ(expected[i][1], actual[i][1]);
        }
    }

    void expect_eq(VecVecReal2 const& expected, VecVecReal2 const& actual)
    {
        ASSERT_EQ(expected.size(), actual.size());
        for (auto i : range(expected.size()))
        {
            EXPECT_EQ(expected[i].size(), actual[i].size());

            for (auto j : range(expected[i].size()))
            {
                EXPECT_EQ(expected[i][j][0], actual[i][j][0]);
                EXPECT_EQ(expected[i][j][1], actual[i][j][1]);
            }
        }
    }

    Tolerance<> t;
};

//---------------------------------------------------------------------------//
/*
 * Test basic configuration with a single "level" of concavity.
 *
 * The starting point is point 5.
 *
 * \verbatim
   2 _______________________ 1
     \                      |
       \                    |
         \                  |
           \                |
           / 3              |
         /                  |
       /                    |
   4 /                      |
     \                      |
    5 \_____________________| 0
   \endverbatim
 */
TEST_F(ConvexHullFinderTest, basic)
{
    VecReal2 p{{0, 0}, {0, 1}, {-1, 1}, {-0.8, 0.5}, {-0.95, 0.2}, {-0.9, 0}};

    // Compare convex hulls using 1--6 points
    expect_eq(VecReal2({p[0], p[1], p[2]}),
              CHF({p[0], p[1], p[2]}, t).make_convex_hull());
    expect_eq(VecReal2({p[0], p[1], p[2], p[3]}),
              CHF({p[0], p[1], p[2], p[3]}, t).make_convex_hull());
    expect_eq(VecReal2({p[0], p[1], p[2], p[4]}),
              CHF({p[0], p[1], p[2], p[4]}, t).make_convex_hull());
    expect_eq(VecReal2({p[0], p[1], p[2], p[4], p[5]}),
              CHF(p, t).make_convex_hull());

    // Compare concave regions using all 6 points
    expect_eq(VecVecReal2({{p[4], p[3], p[2]}}),
              CHF(p, t).calc_concave_regions());
}
//---------------------------------------------------------------------------//
/*
 * Test case where the first point encountered is *not* part of the convex
 * hull.
 *
 * The starting point is point 2.
 * \verbatim
   1 _______________ 0
    |              /
    |    3       /
    |    /\    /
    |  /    \/
    |/      4
    2
    \endverbatim
 */
TEST_F(ConvexHullFinderTest, first_concavity)
{
    VecReal2 p{{0.3, 1}, {-0.9, 1}, {-0.8, 0.4}, {-0.5, 0.7}, {-0.15, 0.5}};
    CHF chf(p, t);
    expect_eq(VecReal2({p[0], p[1], p[2], p[4]}), chf.make_convex_hull());
    expect_eq(VecVecReal2({{p[4], p[3], p[2]}}), chf.calc_concave_regions());
}

//---------------------------------------------------------------------------//
/*
 * Test case where the last point encountered is *not* part of the convex hull.
 *
 * The starting point is point 4.
 * \verbatim
   1 _______________ 0
    |              /
    |____ 3      /
    2     \    /
            \/
            4
   \endverbatim
 */
TEST_F(ConvexHullFinderTest, last_concavity)
{
    VecReal2 p{{0, 0}, {-1, 0}, {-1, -0.5}, {-0.6, -0.5}, {-0.4, -0.8}};
    CHF chf(p, t);
    expect_eq(VecReal2({p[0], p[1], p[2], p[4]}), chf.make_convex_hull());
    expect_eq(VecVecReal2({{p[4], p[3], p[2]}}), chf.calc_concave_regions());
}

//---------------------------------------------------------------------------//
/*
 * Test case with many collinear points, including the first and last points
 * encountered.
 *
 * The starting point is point 7.
 * \verbatim
    2 _______1_______ 0
   3_|              /
     |_____ 5     /
     4    6_\   /  8
             \/
             7
   \endverbatim
 *
 */
TEST_F(ConvexHullFinderTest, collinear)
{
    VecReal2 p{{0, 0},
               {-0.5, 0},
               {-1, 0},
               {-1, -0.2},
               {-1, -0.5},
               {-0.6, -0.5},
               {-0.5, -0.65},
               {-0.4, -0.8},
               {-0.2, -0.4}};
    CHF chf(p, t);
    expect_eq(VecReal2({p[0], p[1], p[2], p[3], p[4], p[7], p[8]}),
              chf.make_convex_hull());
    expect_eq(VecVecReal2({{p[7], p[6], p[5], p[4]}}),
              chf.calc_concave_regions());
}
//---------------------------------------------------------------------------//
/*
 * Test case with a quadruply nested concavity.
 *
 * The starting point is point 9.
 * \verbatim
          7
         /|                 1
        / |                //
       /  |    5  3      / /
      /   |    /\/\    /  /
   8 /    |  /  4   \/   /
     \    |/        2   /
      \   6            /
       \           11 /
        \__________/\/
        9        01  0
   \endverbatim
 */
TEST_F(ConvexHullFinderTest, nested_concavity)
{
    VecReal2 p{{-0.001, 0.001},
               {0.3, 1},
               {-0.15, 0.5},
               {-0.4, 0.7},
               {-0.45, 0.6},
               {-0.5, 0.7},
               {-0.8, 0.4},
               {-0.9, 1.2},
               {-1.2, 0.5},
               {-1, 0},
               {-0.1, 0},
               {-0.05, 0.01}};

    // Test level 0
    CHF chf0(p, t);
    expect_eq(VecReal2({p[0], p[1], p[7], p[8], p[9], p[10]}),
              chf0.make_convex_hull());
    expect_eq(VecVecReal2({{p[7], p[6], p[5], p[4], p[3], p[2], p[1]},
                           {p[0], p[11], p[10]}}),
              chf0.calc_concave_regions());

    // Test level 1
    auto level1_points = chf0.calc_concave_regions();

    CHF chf1a(level1_points[0], t);
    expect_eq(VecReal2({p[7], p[6], p[2], p[1]}), chf1a.make_convex_hull());
    expect_eq(VecVecReal2({{p[2], p[3], p[4], p[5], p[6]}}),
              chf1a.calc_concave_regions());

    CHF chf1b(level1_points[1], t);
    expect_eq(VecReal2({p[0], p[11], p[10]}), chf1b.make_convex_hull());
    expect_eq(VecVecReal2({}), chf1b.calc_concave_regions());

    // Test level 2
    auto level2_points = chf1a.calc_concave_regions()[0];
    CHF chf2(level2_points, t);
    expect_eq(VecReal2({p[2], p[3], p[5], p[6]}), chf2.make_convex_hull());
    expect_eq(VecVecReal2({{p[5], p[4], p[3]}}), chf2.calc_concave_regions());

    // Test level 3
    auto level3_points = chf2.calc_concave_regions()[0];
    CHF chf3(level3_points, t);
    expect_eq(VecReal2({p[5], p[4], p[3]}), chf3.make_convex_hull());
    expect_eq(VecVecReal2({}), chf3.calc_concave_regions());
}

}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
