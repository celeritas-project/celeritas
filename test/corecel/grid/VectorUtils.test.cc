//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/VectorUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/VectorUtils.hh"

#include <vector>

#include "celeritas_test.hh"

constexpr double exact_third = 1.0 / 3.0;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(VectorUtils, linspace)
{
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(linspace(1.23, 4.56, 0), DebugError);
        EXPECT_THROW(linspace(1.23, 4.56, 1), DebugError);
        EXPECT_THROW(linspace(4.56, 1.23, 3), DebugError);
    }

    {
        static double const expected[] = {10, 20};
        EXPECT_VEC_SOFT_EQ(expected, linspace(10, 20, 2));
    }
    {
        static double const expected[] = {10, 12.5, 15, 17.5, 20};
        EXPECT_VEC_SOFT_EQ(expected, linspace(10, 20, 5));
    }
    {
        // Guard against accumulation error
        auto result = linspace(exact_third, 2 * exact_third, 32768);
        ASSERT_EQ(32768, result.size());
        EXPECT_DOUBLE_EQ(exact_third, result.front());
        EXPECT_DOUBLE_EQ(2 * exact_third, result.back());
    }
}

//---------------------------------------------------------------------------//

TEST(VectorUtils, geomspace)
{
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(geomspace(1.23, 4.56, 0), DebugError);
        EXPECT_THROW(geomspace(1.23, 4.56, 1), DebugError);
        EXPECT_THROW(geomspace(4.56, 1.23, 3), DebugError);
    }

    {
        static double const expected[] = {10, 100};
        EXPECT_VEC_SOFT_EQ(expected, geomspace(10, 100, 2));
    }
    {
        static double const expected[] = {2, 4, 8, 16, 32, 64, 128};
        EXPECT_VEC_SOFT_EQ(expected, geomspace(2, 128, 7));
    }
    {
        static double const expected[] = {2,
                                          2.5198420997897,
                                          3.1748021039364,
                                          4,
                                          5.0396841995795,
                                          6.3496042078728,
                                          8};
        EXPECT_VEC_SOFT_EQ(expected, geomspace(2, 8, 7));
    }
    {
        // Guard against accumulation error
        auto result = geomspace(exact_third, 2 * exact_third, 32768);
        ASSERT_EQ(32768, result.size());
        EXPECT_DOUBLE_EQ(exact_third, result.front());
        EXPECT_DOUBLE_EQ(2 * exact_third, result.back());
    }
}

//---------------------------------------------------------------------------//

TEST(VectorUtils, monotonic_nondecreasing)
{
    {
        std::vector<real_type> v{2, 4, 8, 16, 32};
        EXPECT_TRUE(is_monotonic_nondecreasing(make_span(v)));
    }
    {
        std::vector<real_type> v{10, 100, 1000, 1000};
        EXPECT_TRUE(is_monotonic_nondecreasing(make_span(v)));
    }
    {
        std::vector<real_type> v{10, 100, 1000, 100};
        EXPECT_FALSE(is_monotonic_nondecreasing(make_span(v)));
    }
}

TEST(VectorUtils, monotonic_increasing)
{
    {
        std::vector<real_type> v{2, 4, 8, 16, 32};
        EXPECT_TRUE(is_monotonic_increasing(make_span(v)));
    }
    {
        std::vector<real_type> v{10, 100, 1000, 1000};
        EXPECT_FALSE(is_monotonic_increasing(make_span(v)));
    }
    {
        std::vector<real_type> v{1e-16, 0, 1, 2};
        EXPECT_FALSE(is_monotonic_increasing(make_span(v)));
    }
}

TEST(VectorUtils, has_log_spacing)
{
    {
        std::vector<real_type> const v{1e1, 1e2, 1e3, 1e4};
        EXPECT_TRUE(has_log_spacing(make_span(v)));
    }
    {
        std::vector<real_type> const v{1e1, 1e2, 1e3, 1e5};
        EXPECT_FALSE(has_log_spacing(make_span(v)));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
