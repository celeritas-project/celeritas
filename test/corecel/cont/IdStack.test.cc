//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/IdStack.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/IdStack.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(IdStackTest, fixed_size)
{
    Array<int, 3> storage = {0, 0, 0};
    // NOTE: GCC 11.5 fails to compile Span{storage}:
    // > class template placeholder 'celeritas::Span' not permitted in this
    // context
    IdStack stack{Span<int, 3>{storage}};
    struct ExpectedStructSize
    {
        int* data;
        int top;
        size_type size;
        bool empty;
    };
    EXPECT_EQ(sizeof(ExpectedStructSize), sizeof(stack));

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());

    // Push  and pop
    stack.push(42);
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(1, stack.size());
    EXPECT_EQ(42, stack.top());
    stack.pop();

    // Push more
    ASSERT_EQ(0, stack.size());
    stack.push(10);
    stack.push(20);
    stack.push(30);

    EXPECT_EQ(3, stack.size());
    EXPECT_EQ(10, storage[0]);
    EXPECT_EQ(20, storage[1]);
    EXPECT_EQ(30, stack.top());

    EXPECT_EQ(30, stack.top());
    stack.pop();
    EXPECT_EQ(2, stack.size());

    EXPECT_EQ(20, stack.top());
    stack.pop();
    EXPECT_EQ(1, stack.size());

    EXPECT_EQ(10, stack.top());
    stack.pop();
    EXPECT_EQ(0, stack.size());
    EXPECT_TRUE(stack.empty());
}

TEST(IdStackTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    Array<int, 1> storage = {0};
    IdStack<int, 1> stack(make_span(storage));
    EXPECT_EQ(2, stack.capacity());
    // Pop empty should throw
    EXPECT_THROW(stack.pop(), DebugError);

    // Push full should throw
    stack.push(1);
    stack.push(2);
    if constexpr (!CELERITAS_DEBUG)
    {
        // Silence GCC warning
        CELER_UNREACHABLE;
    }
    EXPECT_THROW(stack.push(3), DebugError);
}

TEST(IdStackTest, dynamic_span_construct)
{
    Array<int, 3> storage = {0, 0, 0};
    Span<int> dynamic_span(storage.data(), storage.size());
    IdStack stack(dynamic_span);

    struct ExpectedStructSize
    {
        int* data;
        std::size_t cap;
        int top;
        size_type size;
        bool empty;
    };
    EXPECT_EQ(sizeof(ExpectedStructSize), sizeof(stack));

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());
}

TEST(IdStackTest, different_size_construct)
{
    Array<int, 3> storage = {0, 0, 0};
    IdStack<int, 3, short int> stack(make_span(storage));

    struct ExpectedStructSize
    {
        int* data;
        int top;
        short int size;
        bool empty;
    };
    EXPECT_EQ(sizeof(ExpectedStructSize), sizeof(stack));

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
