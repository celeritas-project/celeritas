//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/IllinoisRootFinder.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/ConvexHullFinder.hh"

#include <cmath>
#include <functional>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

class ConvexHullFinderTest : public ::celeritas::test::Test
{
  public:
    using Real2 = std::array<double, 2>;
    using VecReal2 = std::vector<Real2>;
    using VecIdx = std::vector<size_type>;
};

//---------------------------------------------------------------------------//
// Test basic configuration with a single level of concavity
//    1 _______________________ 2
//     |                      /
//     |                    /
//     |                  /
//     |                /
//     |              3 \
//     |                  \
//     |                    \
//     |                      \ 4
//     |                      /
//   0 |_____________________/ 5
//---------------------------------------------------------------------------//

TEST_F(ConvexHullFinderTest, basic)
{
    std::vector<Real2> points{
        {0, 0}, {0, 1}, {1, 1}, {0.8, 0.5}, {0.95, 0.2}, {0.9, 0}};
    ConvexHullFinder<Real2> chf(points);

    // Start from index 0
    EXPECT_VEC_EQ(VecIdx({0}), chf({0, 1}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1}), chf({0, 2}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1, 2}), chf({0, 3}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1, 2, 3}), chf({0, 4}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1, 2, 4}), chf({0, 5}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1, 2, 4, 5}), chf({0, 6}).convex_hull);

    // Start from index 1; now our starting point (the point with the min y)
    // changes.
    EXPECT_VEC_EQ(VecIdx({1}), chf({1, 2}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({1, 2}), chf({1, 3}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({1, 2, 3}), chf({1, 4}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({4, 1, 2}), chf({1, 5}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({5, 1, 2, 4}), chf({1, 6}).convex_hull);

    // Specify rotate the points, such that we start counting from 1
    std::vector<Real2> points2{
        {0, 1}, {1, 1}, {0.8, 0.5}, {0.95, 0.2}, {0.9, 0}, {0, 0}};
    ConvexHullFinder<Real2> chf2(points2);

    EXPECT_VEC_EQ(VecIdx({0}), chf2({0, 1}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1}), chf2({0, 2}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({0, 1, 2}), chf2({0, 3}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({3, 0, 1}), chf2({0, 4}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({4, 0, 1, 3}), chf2({0, 5}).convex_hull);
    EXPECT_VEC_EQ(VecIdx({4, 5, 0, 1, 3}), chf2({0, 6}).convex_hull);
}

//---------------------------------------------------------------------------//
// Test with a triply nested concavity.
//
//                     7
//   1                 |\
//   \\                | \
//    \ \      3  5    |  \
//     \  \    /\/\    |   \
//      \   \/   4  \  |    \ 8
//       \   2        \|    /
//        \            6   /
//         \              /
//          \____________/
//          0            9
//---------------------------------------------------------------------------//
TEST_F(ConvexHullFinderTest, nested_concavity)
{
    std::vector<Real2> points{{0, 0},
                              {-0.3, 1},
                              {0.15, 0.5},
                              {0.4, 0.7},
                              {0.45, 0.6},
                              {0.5, 0.7},
                              {0.8, 0.4},
                              {0.9, 1.2},
                              {1.2, 0.5},
                              {1, 0}};

    ConvexHullFinder<Real2> chf(points);
    // Start from index 0
    EXPECT_VEC_EQ(VecIdx({0, 1, 7, 8, 9}), chf({0, 10}).convex_hull);
}

}  // namespace test
}  // namespace celeritas
