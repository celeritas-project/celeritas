//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/MiniStack.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/MiniStack.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(MiniStackTest, fixed_size)
{
    Array<int, 3> storage;
    MiniStack stack(Span{storage});

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(3, stack.capacity());

    // Push  and pop
    stack.push(42);
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(1, stack.size());
    EXPECT_EQ(42, stack.pop());

    // Push more
    ASSERT_EQ(0, stack.size());
    stack.push(10);
    stack.push(20);
    stack.push(30);

    EXPECT_EQ(3, stack.size());
    EXPECT_EQ(10, storage[0]);
    EXPECT_EQ(20, storage[1]);
    EXPECT_EQ(30, storage[2]);

    EXPECT_EQ(30, stack.pop());
    EXPECT_EQ(2, stack.size());

    EXPECT_EQ(20, stack.pop());
    EXPECT_EQ(1, stack.size());

    EXPECT_EQ(10, stack.pop());
    EXPECT_EQ(0, stack.size());
    EXPECT_TRUE(stack.empty());
}

TEST(MiniStackTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    Array<int, 1> storage = {0};
    MiniStack<int> stack(Span{storage});
    EXPECT_EQ(1, stack.capacity());
    // Pop empty should throw
    EXPECT_THROW(stack.pop(), DebugError);

    // Push full should throw
    stack.push(1);
    EXPECT_THROW(stack.push(2), DebugError);
}

TEST(MiniStackTest, dynamic_span_construct)
{
    Array<int, 3> storage = {0, 0, 0};
    Span<int> dynamic_span(storage.data(), storage.size());
    MiniStack<int> stack(dynamic_span);

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(3, stack.capacity());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
