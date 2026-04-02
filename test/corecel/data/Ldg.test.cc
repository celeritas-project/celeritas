//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgIterator.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <numeric>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/LdgSpan.hh"
#include "corecel/cont/Span.hh"
#include "corecel/cont/detail/LdgIterator.hh"
#include "corecel/data/LdgRefWrapper.hh"
#include "corecel/math/Quantity.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct DozenUnit
{
    static constexpr int value() { return 12; }
    static constexpr char const* label() { return "dozen"; }
};

using Dozen = Quantity<DozenUnit, int>;

//---------------------------------------------------------------------------//
using LdgRefWrapperTest = Test;

TEST_F(LdgRefWrapperTest, quantity)
{
    static Dozen const eggs[] = {Dozen{1}, Dozen{3}, Dozen{5}};
    LdgSpan<Dozen const> view{eggs};
    ASSERT_EQ(3, view.size());
    EXPECT_EQ(dynamic_extent, view.extent);

    EXPECT_TRUE(
        (std::is_same_v<decltype(view.back()), LdgRefWrapper<Dozen const>>));

    auto implicitly_converted = view.back() * 2;
    EXPECT_TRUE((std::is_same_v<decltype(implicitly_converted), Dozen>));
    EXPECT_EQ(Dozen{10}, implicitly_converted);
}

//---------------------------------------------------------------------------//

using detail::LdgIterator;

using LdgIteratorTest = Test;

TEST_F(LdgIteratorTest, arithmetic_t)
{
    using VecInt = std::vector<int>;
    using RefInt = LdgRefWrapper<int const>;
    VecInt const some_data = {1, 2, 3, 4};
    auto n = some_data.size();
    auto start = some_data.begin();
    auto end = some_data.end();

    auto ldg_start = LdgIterator(some_data.data());
    auto ldg_end = LdgIterator(some_data.data() + n);
    LdgIterator ctad_itr{some_data.data()};
    EXPECT_TRUE((std::is_same_v<decltype(ctad_itr), decltype(ldg_start)>));

    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE((std::is_same_v<ptr_type, int const*>));

    EXPECT_TRUE(ldg_start);
    EXPECT_NE(ldg_start, nullptr);
    EXPECT_NE(nullptr, ldg_start);
    EXPECT_EQ(std::accumulate(start, end, 0),
              std::accumulate(ldg_start, ldg_end, 0));
    EXPECT_EQ(static_cast<ptr_type>(ldg_start), some_data.data());

    EXPECT_TRUE((std::is_same_v<decltype(*ldg_start), RefInt>));
    EXPECT_EQ(1 + 2, *ldg_start + 2);  // test implicit conversion from wrapper
    EXPECT_EQ(*ldg_start++, 1);
    EXPECT_EQ(*ldg_start--, 2);
    EXPECT_EQ(*++ldg_start, 2);
    EXPECT_EQ(*--ldg_start, 1);
    EXPECT_EQ(ldg_start[n - 1], some_data.back());
    EXPECT_GT(ldg_end, ldg_start);
    auto ldg_start_copy = ldg_start;
    EXPECT_EQ(ldg_start, ldg_start_copy);
    ldg_start += n;
    EXPECT_NE(ldg_start, ldg_start_copy);
    EXPECT_EQ(ldg_start, ldg_end);
    ldg_end -= n;
    EXPECT_EQ(ldg_end, ldg_start_copy);
    std::swap(ldg_start, ldg_end);
    EXPECT_EQ(ldg_start, ldg_start_copy);
    EXPECT_EQ(ldg_end, ldg_start + n);
    EXPECT_EQ(ldg_end, n + ldg_start);
    EXPECT_EQ(ldg_end - n, ldg_start);
    EXPECT_EQ(ldg_end - ldg_start, n);
    ldg_end = ldg_start;
    EXPECT_EQ(ldg_end, ldg_start);
    auto ldg_nullptr = LdgIterator<int const>{nullptr};
    EXPECT_EQ(ldg_nullptr, nullptr);
    EXPECT_EQ(nullptr, ldg_nullptr);
    EXPECT_FALSE(ldg_nullptr);
}

TEST_F(LdgIteratorTest, opaqueid_t)
{
    using TestId = OpaqueId<struct LdgIteratorOpaqueIdTest_>;
    using VecId = std::vector<TestId>;
    VecId const some_data = {TestId{1}, TestId{2}, TestId{3}, TestId{4}};
    auto n = some_data.size();
    auto ldg_start = LdgIterator(some_data.data());
    auto ldg_end = LdgIterator(some_data.data() + n);
    LdgIterator ctad_itr{some_data.data()};
    EXPECT_TRUE((std::is_same_v<decltype(ctad_itr), decltype(ldg_start)>));
    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE((std::is_same_v<ptr_type, TestId const*>));
    EXPECT_TRUE(ldg_start);
    EXPECT_NE(ldg_start, nullptr);
    EXPECT_NE(nullptr, ldg_start);
    EXPECT_EQ(static_cast<ptr_type>(ldg_start), some_data.data());
    EXPECT_EQ(ldg_start->unchecked_get(), 1);
    EXPECT_EQ(*ldg_start++, TestId{1});
    EXPECT_EQ(*ldg_start--, TestId{2});
    EXPECT_EQ(*++ldg_start, TestId{2});
    EXPECT_EQ(*--ldg_start, TestId{1});
    EXPECT_EQ(ldg_start[n - 1], some_data.back());
    EXPECT_GT(ldg_end, ldg_start);
    auto ldg_start_copy = ldg_start;
    EXPECT_EQ(ldg_start, ldg_start_copy);
    ldg_start += n;
    EXPECT_NE(ldg_start, ldg_start_copy);
    EXPECT_EQ(ldg_start, ldg_end);
    ldg_end -= n;
    EXPECT_EQ(ldg_end, ldg_start_copy);
    std::swap(ldg_start, ldg_end);
    EXPECT_EQ(ldg_start, ldg_start_copy);
    EXPECT_EQ(ldg_end, ldg_start + n);
    EXPECT_EQ(ldg_end, n + ldg_start);
    EXPECT_EQ(ldg_end - n, ldg_start);
    EXPECT_EQ(ldg_end - ldg_start, n);
    ldg_end = ldg_start;
    EXPECT_EQ(ldg_end, ldg_start);
    auto ldg_nullptr = LdgIterator<int const>{nullptr};
    EXPECT_EQ(ldg_nullptr, nullptr);
    EXPECT_EQ(nullptr, ldg_nullptr);
    EXPECT_FALSE(ldg_nullptr);
}

TEST_F(LdgIteratorTest, byte_t)
{
    using VecByte = std::vector<std::byte>;
    VecByte const some_data
        = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    auto n = some_data.size();
    auto ldg_start = LdgIterator(some_data.data());
    auto ldg_end = LdgIterator(some_data.data() + n);
    LdgIterator ctad_itr{some_data.data()};
    EXPECT_TRUE((std::is_same_v<decltype(ctad_itr), decltype(ldg_start)>));
    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE((std::is_same_v<ptr_type, std::byte const*>));
    EXPECT_TRUE(ldg_start);
    EXPECT_NE(ldg_start, nullptr);
    EXPECT_NE(nullptr, ldg_start);
    EXPECT_EQ(static_cast<ptr_type>(ldg_start), some_data.data());
    EXPECT_EQ(*ldg_start++, std::byte{1});
    EXPECT_EQ(*ldg_start--, std::byte{2});
    EXPECT_EQ(*++ldg_start, std::byte{2});
    EXPECT_EQ(*--ldg_start, std::byte{1});
    EXPECT_EQ(ldg_start[n - 1], some_data.back());
    EXPECT_GT(ldg_end, ldg_start);
    auto ldg_start_copy = ldg_start;
    EXPECT_EQ(ldg_start, ldg_start_copy);
    ldg_start += n;
    EXPECT_NE(ldg_start, ldg_start_copy);
    EXPECT_EQ(ldg_start, ldg_end);
    ldg_end -= n;
    EXPECT_EQ(ldg_end, ldg_start_copy);
    std::swap(ldg_start, ldg_end);
    EXPECT_EQ(ldg_start, ldg_start_copy);
    EXPECT_EQ(ldg_end, ldg_start + n);
    EXPECT_EQ(ldg_end, n + ldg_start);
    EXPECT_EQ(ldg_end - n, ldg_start);
    EXPECT_EQ(ldg_end - ldg_start, n);
    ldg_end = ldg_start;
    EXPECT_EQ(ldg_end, ldg_start);
    auto ldg_nullptr = LdgIterator<int const>{nullptr};
    EXPECT_EQ(ldg_nullptr, nullptr);
    EXPECT_EQ(nullptr, ldg_nullptr);
    EXPECT_FALSE(ldg_nullptr);
}

TEST_F(LdgIteratorTest, enum_class)
{
    enum class Color
    {
        r,
        g,
        b
    };
    static Color colors[] = {Color::r, Color::b, Color::g};

    LdgIterator start(std::begin(colors));
    LdgIterator end(std::end(colors));

    EXPECT_EQ(3, end - start);
    EXPECT_EQ(Color::r, *start);
    EXPECT_EQ(Color::g, *(end - 1));
}

#ifdef CELERITAS_SHOULD_NOT_COMPILE
// Note that this will fail to compile due to the invalid type
TEST_F(LdgIteratorTest, invalid_type)
{
    std::pair<int, int> ints;

    LdgIterator start{&ints};
    EXPECT_EQ(&ints, &(*start));
}
#endif

//---------------------------------------------------------------------------//
using LdgSpanTest = Test;

TEST_F(LdgSpanTest, pod)
{
    using LdgInt = LdgRefWrapper<int const>;
    int local_data[] = {123, 456, 789};
    Span<int> mutable_span(local_data);
    EXPECT_TRUE((std::is_same_v<decltype(mutable_span[0]), int&>));
    Span<LdgInt> ldg_span(mutable_span);
    Span<LdgInt> local_span(local_data);
    EXPECT_TRUE(
        (std::is_same_v<typename Span<LdgInt>::element_type, int const>));
    EXPECT_TRUE((std::is_same_v<decltype(local_span.data()), int const*>));
    EXPECT_TRUE((std::is_same_v<decltype(local_span.front()), LdgInt>));
    EXPECT_TRUE((std::is_same_v<decltype(local_span.back()), LdgInt>));
    EXPECT_TRUE((std::is_same_v<decltype(local_span[0]), LdgInt>));
    EXPECT_TRUE((
        std::is_same_v<decltype(local_span.begin()), LdgIterator<int const>>));
    EXPECT_TRUE(
        (std::is_same_v<decltype(local_span.end()), LdgIterator<int const>>));

    EXPECT_EQ(local_span.first(2).back(), 456);
    EXPECT_TRUE(
        (std::is_same_v<decltype(local_span), decltype(local_span.first(2))>));
    EXPECT_EQ(local_span.subspan(1, 1)[1], 789);

    auto begin = local_span.begin();
    EXPECT_EQ(*begin++, 123);
    EXPECT_EQ(*begin++, 456);
    EXPECT_EQ(*begin++, 789);
    EXPECT_EQ(begin, local_span.end());
    EXPECT_EQ(local_span[2], 789);
    EXPECT_EQ(local_span.end()[-3], 123);
}

TEST_F(LdgSpanTest, opaque_id)
{
    using TestId = OpaqueId<struct SpanTestLdgOpaqueId_>;
    using LdgId = LdgRefWrapper<TestId const>;

    TestId local_data[] = {TestId{123}, TestId{456}, TestId{789}};
    Span<TestId> mutable_span(local_data);
    EXPECT_TRUE((std::is_same_v<decltype(mutable_span[0]), TestId&>));
    Span<LdgId> ldg_span(mutable_span);
    Span<LdgId> s(local_data);
    EXPECT_TRUE(
        (std::is_same_v<typename Span<LdgId>::element_type, TestId const>));
    EXPECT_TRUE((std::is_same_v<decltype(s.data()), TestId const*>));
    EXPECT_TRUE((std::is_same_v<decltype(s.front()), LdgId>));
    EXPECT_TRUE((std::is_same_v<decltype(s.back()), LdgId>));
    EXPECT_TRUE((std::is_same_v<decltype(s[0]), LdgId>));
    EXPECT_TRUE(
        (std::is_same_v<decltype(s.begin()), LdgIterator<TestId const>>));
    EXPECT_TRUE((std::is_same_v<decltype(s.end()), LdgIterator<TestId const>>));

    EXPECT_EQ(s.first(2).back(), TestId{456});
    EXPECT_TRUE((std::is_same_v<decltype(s), decltype(s.first(2))>));
    EXPECT_EQ(s.subspan(1, 1)[1], TestId{789});

    auto begin = s.begin();
    EXPECT_EQ(*begin++, TestId{123});
    EXPECT_EQ(*begin++, TestId{456});
    EXPECT_EQ(*begin++, TestId{789});
    EXPECT_EQ(begin, s.end());
    EXPECT_EQ(s[2], TestId{789});
    EXPECT_EQ(s.end()[-3], TestId{123});
}

//---------------------------------------------------------------------------//
using LdgMemberTest = Test;

TEST_F(LdgMemberTest, two_arg_ldg)
{
    using TestId = OpaqueId<struct LdgMemberOpaqueIdTest_>;

    struct Node
    {
        int value;
        TestId id;
    };

    static Node const nodes[] = {{3, TestId{7}}, {5, TestId{2}}};

    EXPECT_EQ(3, ldg(nodes[0], &Node::value));
    EXPECT_EQ(5, ldg(nodes[1], &Node::value));
    EXPECT_EQ(TestId{7}, ldg(nodes[0], &Node::id));
    EXPECT_EQ(TestId{2}, ldg(nodes[1], &Node::id));
}

TEST_F(LdgMemberTest, ldg_member_callable)
{
    using TestId = OpaqueId<struct LdgMemberCallableTest_>;

    struct Node
    {
        int value;
        TestId id;
    };

    static Node const nodes[] = {{3, TestId{7}}, {5, TestId{2}}};

    auto load_value = LdgMember{&Node::value};
    auto load_id = LdgMember{&Node::id};

    EXPECT_TRUE((std::is_same_v<decltype(load_value), LdgMember<Node, int>>));
    EXPECT_TRUE((std::is_same_v<decltype(load_id), LdgMember<Node, TestId>>));

    EXPECT_EQ(3, load_value(nodes[0]));
    EXPECT_EQ(5, load_value(nodes[1]));
    EXPECT_EQ(TestId{7}, load_id(nodes[0]));
    EXPECT_EQ(TestId{2}, load_id(nodes[1]));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
