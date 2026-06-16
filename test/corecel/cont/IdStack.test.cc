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
using TestId = OpaqueId<struct Test_>;

TEST(IdStackTest, fixed_size)
{
    Array<TestId, 3> storage = {TestId{}, TestId{}, TestId{}};
    // NOTE: GCC 11.5 fails to compile Span{storage}:
    // > class template placeholder 'celeritas::Span' not permitted in this
    // context
    IdStack stack{Span<TestId, 3>{storage}};
    struct ExpectedStructSize
    {
        TestId* data;
        TestId top;
        size_type size;
    };
    EXPECT_EQ(sizeof(ExpectedStructSize), sizeof(stack));

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());

    // Push  and pop
    stack.push(TestId{42});
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(1, stack.size());
    EXPECT_EQ(TestId{42}, stack.top());
    stack.pop();

    // Push more
    ASSERT_EQ(0, stack.size());
    stack.push(TestId{10});
    stack.push(TestId{20});
    stack.push(TestId{30});

    EXPECT_EQ(3, stack.size());
    EXPECT_EQ(TestId{10}, storage[0]);
    EXPECT_EQ(TestId{20}, storage[1]);
    EXPECT_EQ(TestId{30}, stack.top());

    EXPECT_EQ(TestId{30}, stack.top());
    stack.pop();
    EXPECT_EQ(2, stack.size());

    EXPECT_EQ(TestId{20}, stack.top());
    stack.pop();
    EXPECT_EQ(1, stack.size());

    EXPECT_EQ(TestId{10}, stack.top());
    stack.pop();
    EXPECT_EQ(0, stack.size());
    EXPECT_TRUE(stack.empty());
}

TEST(IdStackTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    Array<TestId, 1> storage = {TestId{0}};
    IdStack<TestId, 1> stack(make_span(storage));
    EXPECT_EQ(2, stack.capacity());
    // Pop empty should throw
    EXPECT_THROW(stack.pop(), DebugError);

    // Push full should throw
    stack.push(TestId{1});
    stack.push(TestId{2});
    if constexpr (!CELERITAS_DEBUG)
    {
        // Silence GCC warning
        CELER_UNREACHABLE;
    }
    EXPECT_THROW(stack.push(TestId{3}), DebugError);
}

TEST(IdStackTest, dynamic_span_construct)
{
    Array<TestId, 3> storage;
    Span<TestId> dynamic_span(storage.data(), storage.size());
    IdStack stack(dynamic_span);

    struct ExpectedStructSize
    {
        TestId* data;
        std::size_t cap;
        TestId top;
        size_type size;
    };
    EXPECT_EQ(sizeof(ExpectedStructSize), sizeof(stack));

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());
}

TEST(IdStackTest, different_size_construct)
{
    Array<TestId, 3> storage;
    IdStack<TestId, 3, short int> stack(make_span(storage));

    struct ExpectedStructSize
    {
        TestId* data;
        TestId top;
        short int size;
    };
    EXPECT_EQ(sizeof(ExpectedStructSize), sizeof(stack));

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
