//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VectorUtils.test.cc
//---------------------------------------------------------------------------//
#include "base/VectorUtils.hh"

#include "celeritas_test.hh"

using celeritas::DebugError;
using celeritas::linspace;
using celeritas::real_type;

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//

struct Moveable
{
    int  value;
    int* counter;

    Moveable(int v, int* c) : value(v), counter(c) { CELER_EXPECT(counter); }

    Moveable(Moveable&& rhs) noexcept : value(rhs.value), counter(rhs.counter)
    {
        ++(*counter);
    }

    Moveable& operator=(Moveable&& rhs) noexcept
    {
        value   = rhs.value;
        counter = rhs.counter;
        ++(*counter);
        return *this;
    }

    // Delete copy and copy assign
    Moveable(const Moveable& rhs) = delete;
    Moveable& operator=(const Moveable& rhs) = delete;
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class VectorUtilsTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(VectorUtilsTest, extend)
{
    std::vector<int> dst;
    dst.reserve(4);
    {
        // Span of same type
        const int src[] = {3};
        auto inserted   = celeritas::extend(celeritas::make_span(src), &dst);
        EXPECT_EQ(1, inserted.size());
        EXPECT_EQ(dst.data(), inserted.data());
    }
    {
        // Span with type conversion
        const unsigned long src[] = {2};
        auto inserted = celeritas::extend(celeritas::make_span(src), &dst);
        EXPECT_EQ(1, inserted.size());
        EXPECT_EQ(dst.data() + 1, inserted.data());
    }
    {
        // Vector of same type
        auto inserted = celeritas::extend(std::vector<int>({1, 0}), &dst);
        EXPECT_EQ(2, inserted.size());
        EXPECT_EQ(dst.data() + 2, inserted.data());
    }

    const int expected_dst[] = {3, 2, 1, 0};
    EXPECT_VEC_EQ(expected_dst, dst);
}

TEST_F(VectorUtilsTest, move_extend)
{
    using VecMov = std::vector<Moveable>;
    VecMov dst;
    int    ctr = 0;

    VecMov src;
    src.emplace_back(1, &ctr);
    src.emplace_back(2, &ctr);
    ctr = 0;

    // NOTE: the move_extend function intentionally will not compile without
    // the std::move -- the caller needs to pass an rvalue.
    celeritas::move_extend(std::move(src), &dst);
    EXPECT_EQ(2, ctr);
    EXPECT_EQ(0, src.size());
    ASSERT_EQ(2, dst.size());
    EXPECT_EQ(1, dst[0].value);
    EXPECT_EQ(2, dst[1].value);
}

TEST_F(VectorUtilsTest, TEST_IF_CELERITAS_DEBUG(error_checking))
{
    std::vector<int> dst;
    dst.reserve(3);
    EXPECT_THROW(celeritas::extend(std::vector<int>(4), &dst),
                 celeritas::DebugError);
    EXPECT_NO_THROW(celeritas::extend(std::vector<int>(3), &dst));
    EXPECT_THROW(celeritas::extend(std::vector<int>(1), &dst),
                 celeritas::DebugError);
}

TEST(VectorUtils, linspace)
{
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(linspace(1.23, 4.56, 0), DebugError);
        EXPECT_THROW(linspace(1.23, 4.56, 1), DebugError);
    }

    {
        auto result = linspace(10, 20, 2);

        static const real_type expected[] = {10, 20};
        EXPECT_VEC_SOFT_EQ(expected, result);
    }
    {
        auto result = linspace(10, 20, 5);

        static const real_type expected[] = {10, 12.5, 15, 17.5, 20};
        EXPECT_VEC_SOFT_EQ(expected, result);
    }
    {
        // Guard against accumulation error
        const real_type exact_third = 1.0 / 3.0;
        auto            result = linspace(exact_third, 2 * exact_third, 32768);
        ASSERT_EQ(32768, result.size());
        if (sizeof(real_type) == sizeof(double))
        {
            EXPECT_DOUBLE_EQ(exact_third, result.front());
            EXPECT_DOUBLE_EQ(2 * exact_third, result.back());
        }
    }
}
