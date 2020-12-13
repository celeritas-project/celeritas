//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Span.test.cc
//---------------------------------------------------------------------------//
#include "base/Span.hh"

#include <type_traits>
#include "celeritas_test.hh"

using celeritas::span;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SpanTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SpanTest, fixed_size_zero)
{
    span<int, 0> empty_span;
    EXPECT_EQ(nullptr, empty_span.data());
    EXPECT_EQ(0, empty_span.size());
    EXPECT_TRUE(empty_span.empty());
    EXPECT_EQ(sizeof(int*), sizeof(empty_span));

    auto templ_subspan = empty_span.subspan<0, 0>();
    EXPECT_TRUE(
        (std::is_same<decltype(empty_span), decltype(templ_subspan)>::value));

    // Test copy constructor
    span<int, 0> other_span{empty_span};
    EXPECT_EQ(nullptr, other_span.data());
    EXPECT_EQ(0, other_span.size());
    empty_span = other_span;

    // Test dynamic conversion
    span<int> dynamic{empty_span};
    EXPECT_EQ(nullptr, dynamic.data());
    EXPECT_EQ(0, dynamic.size());

    // Test type conversion
    span<const int, 0> const_span{empty_span};
    EXPECT_EQ(0, const_span.size());

    // Test type conversion
    span<const int> const_dynamic{empty_span};
    EXPECT_EQ(0, const_dynamic.size());

    // Test pointer constructor
    span<int, 0> ptr_span(empty_span.begin(), empty_span.end());
    EXPECT_EQ(0, ptr_span.size());
}

TEST_F(SpanTest, fixed_size)
{
    int          local_data[] = {123, 456};
    span<int, 2> local_span(local_data);
    EXPECT_EQ(sizeof(int*), sizeof(local_span));

    EXPECT_EQ(local_data, local_span.begin());
    EXPECT_EQ(local_data + 2, local_span.end());
    EXPECT_EQ(local_data, local_span.data());
    EXPECT_EQ(2, local_span.size());
    EXPECT_EQ(2 * sizeof(int), local_span.size_bytes());
    EXPECT_FALSE(local_span.empty());
    EXPECT_EQ(&local_data[1], &(local_span[1]));
    EXPECT_EQ(123, local_span.front());
    EXPECT_EQ(456, local_span.back());

    auto templ_subspan = local_span.subspan<1>();
    EXPECT_TRUE((std::is_same<span<int, 1>, decltype(templ_subspan)>::value));
    EXPECT_EQ(local_data + 1, templ_subspan.data());

    auto func_subspan = local_span.subspan(1, 1);
    EXPECT_EQ(1, func_subspan.size());
    EXPECT_EQ(local_data + 1, func_subspan.data());

    // Test copy constructor
    span<int, 2> other_span{local_span};
    EXPECT_EQ(local_data, other_span.data());
    EXPECT_EQ(2, other_span.size());
    local_span = other_span;

    // Test dynamic conversion
    span<int> dynamic{local_span};
    EXPECT_EQ(local_data, dynamic.data());
    EXPECT_EQ(2, dynamic.size());

    // Test type conversion
    span<const int, 2> const_span{local_span};
    EXPECT_EQ(local_data, const_span.data());
    EXPECT_EQ(2, const_span.size());

    // Test type conversion
    span<const int> const_dynamic{local_span};
    EXPECT_EQ(local_data, const_dynamic.data());
    EXPECT_EQ(2, const_dynamic.size());

    // Test pointer constructor
    span<int, 2> ptr_span(local_data, local_data + 2);
    EXPECT_EQ(local_data, ptr_span.data());
}

TEST_F(SpanTest, dynamic_size)
{
    int       local_data[] = {123, 456, 789};
    span<int> local_span(local_data);
    EXPECT_EQ(sizeof(int*) + sizeof(std::size_t), sizeof(local_span));

    EXPECT_EQ(local_data, local_span.begin());
    EXPECT_EQ(local_data + 3, local_span.end());
    EXPECT_EQ(local_data, local_span.data());
    EXPECT_EQ(3, local_span.size());
    EXPECT_EQ(3 * sizeof(int), local_span.size_bytes());
    EXPECT_FALSE(local_span.empty());
    EXPECT_EQ(&local_data[1], &(local_span[1]));
    EXPECT_EQ(123, local_span.front());
    EXPECT_EQ(789, local_span.back());

    {
        auto templ_subspan = local_span.subspan<1>();
        EXPECT_TRUE((std::is_same<span<int>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data + 1, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.subspan<1, 2>();
        EXPECT_TRUE(
            (std::is_same<span<int, 2>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data + 1, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }

    auto func_subspan = local_span.subspan(1);
    EXPECT_EQ(local_data + 1, func_subspan.data());
    EXPECT_EQ(2, func_subspan.size());

    // Test copy constructor
    span<int> other_span{local_span};
    EXPECT_EQ(local_data, other_span.data());
    EXPECT_EQ(3, other_span.size());
    local_span = other_span;

    // Test dynamic conversion
    span<int> dynamic{local_span};
    EXPECT_EQ(local_data, dynamic.data());
    EXPECT_EQ(3, dynamic.size());

    // Test type conversion
    span<const int> const_dynamic{local_span};
    EXPECT_EQ(local_data, const_dynamic.data());
    EXPECT_EQ(3, const_dynamic.size());

    // Test type and size conversion
    span<const int, 3> const_span{local_span};
    EXPECT_EQ(local_data, const_span.data());
    EXPECT_EQ(3, const_span.size());

    // Test pointer constructor
    span<int> ptr_span(local_data, local_data + 3);
    EXPECT_EQ(local_data, ptr_span.data());
}
