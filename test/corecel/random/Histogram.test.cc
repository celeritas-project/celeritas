//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Histogram.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/Histogram.hh"

#include <vector>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
TEST(HistogramTest, bin)
{
    using VecDbl = std::vector<double>;
    using VecCount = std::vector<size_type>;
    {
        // All on the leftmost bin edge
        VecDbl values(100, 0);
        Histogram bin(10, {0, 1});
        bin(values);

        static unsigned int const expected_counts[]
            = {100u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        static double const expected_density[]
            = {10, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT_VEC_EQ(expected_counts, bin.counts());
        EXPECT_VEC_SOFT_EQ(expected_density, bin.calc_density());
    }
    {
        // All on the rightmost bin edge
        VecDbl values(100, 1);
        Histogram bin(10, {0, 1});
        bin(values);

        static unsigned int const expected_counts[]
            = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 100u};
        static double const expected_density[]
            = {0, 0, 0, 0, 0, 0, 0, 0, 0, 10};
        EXPECT_VEC_EQ(expected_counts, bin.counts());
        EXPECT_VEC_SOFT_EQ(expected_density, bin.calc_density());
    }
    {
        VecDbl values{16, 18, 20, 22, 24, 26, 28, 30, 32};
        Histogram bin(8, {16, 32});
        bin(values);

        static unsigned int const expected_counts[]
            = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u};
        static double const expected_density[] = {
            1.0 / 18,
            1.0 / 18,
            1.0 / 18,
            1.0 / 18,
            1.0 / 18,
            1.0 / 18,
            1.0 / 18,
            1.0 / 9,
        };
        EXPECT_VEC_EQ(expected_counts, bin.counts());
        EXPECT_VEC_SOFT_EQ(expected_density, bin.calc_density());

        bin(32.00000000000001);
        EXPECT_EQ(1, bin.overflow());
        EXPECT_EQ(32.00000000000001, bin.max());

        bin(-1);
        bin(33);
        EXPECT_EQ(1, bin.underflow());
        EXPECT_EQ(2, bin.overflow());
        EXPECT_EQ(-1, bin.min());
        EXPECT_EQ(33, bin.max());
        EXPECT_VEC_EQ(expected_counts, bin.counts());
        EXPECT_VEC_SOFT_EQ(expected_density, bin.calc_density());

        bin(18);
        bin(31.99999999999999);
        EXPECT_VEC_EQ(VecCount({1u, 2u, 1u, 1u, 1u, 1u, 1u, 3u}), bin.counts());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
