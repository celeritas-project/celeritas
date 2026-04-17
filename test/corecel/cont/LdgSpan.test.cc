//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LdgSpan.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/LdgSpan.hh"

#include <algorithm>
#include <numeric>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/detail/LdgSpanImpl.hh"
#include "corecel/math/Quantity.hh"

#include "celeritas_test.hh"

using std::is_same_v;
struct DozenUnit
{
    static constexpr int value() { return 12; }
    static constexpr char const* label() { return "dozen"; }
};

using Dozen = celeritas::Quantity<DozenUnit, int>;

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
using LdgWrapperTest = celeritas::test::Test;

TEST_F(LdgWrapperTest, equality)
{
    // Comparing two LdgWrapper instances was previously ambiguous because
    // both operator==(LdgWrapper, type) and operator==(type, LdgWrapper)
    // matched via implicit conversion, causing a compile error when
    // std::equal is used over two LdgSpans.
    static double const vals[] = {1.0, 2.0};
    LdgWrapper<double const> w1{vals[0]};
    LdgWrapper<double const> w2{vals[0]};
    LdgWrapper<double const> w3{vals[1]};
    EXPECT_EQ(w1, w2);  // LdgWrapper == LdgWrapper (was ambiguous)
    EXPECT_NE(w1, w3);  // LdgWrapper != LdgWrapper (was ambiguous)
    EXPECT_EQ(w1, 1.0);  // LdgWrapper == type
    EXPECT_EQ(1.0, w2);  // type == LdgWrapper

    // std::equal triggers LdgWrapper == LdgWrapper comparison internally
    static double const same_vals[] = {1.0, 2.0};
    auto span1 = make_ldg_span(vals);
    auto span2 = make_ldg_span(same_vals);
    EXPECT_TRUE(std::equal(span1.begin(), span1.end(), span2.begin()));
    static double const diff_vals[] = {1.0, 3.0};
    LdgSpan<double const> span3{diff_vals};
    EXPECT_FALSE(std::equal(span1.begin(), span1.end(), span3.begin()));
}

TEST_F(LdgWrapperTest, quantity)
{
    Dozen const eggs{3};
    Dozen bacon{2};
    LdgWrapper eggs_w{eggs};
    EXPECT_TRUE((is_same_v<decltype(eggs_w), LdgWrapper<Dozen const>>));
    LdgWrapper bacon_w{bacon};
    EXPECT_TRUE((is_same_v<decltype(bacon_w), LdgWrapper<Dozen const>>));

    auto implicitly_converted = eggs_w * 2;
    EXPECT_TRUE((is_same_v<decltype(implicitly_converted), Dozen>));
    EXPECT_EQ(Dozen{6}, implicitly_converted);
}

//---------------------------------------------------------------------------//

using LdgIteratorTest = celeritas::test::Test;

TEST_F(LdgIteratorTest, arithmetic_t)
{
    using VecInt = std::vector<int>;
    using RefInt = LdgWrapper<int const>;
    VecInt const some_data = {1, 2, 3, 4};
    auto n = some_data.size();
    auto start = some_data.begin();
    auto end = some_data.end();

    auto ldg_start = LdgIterator(some_data.data());
    auto ldg_end = LdgIterator(some_data.data() + n);
    LdgIterator ctad_itr{some_data.data()};
    EXPECT_TRUE((is_same_v<decltype(ctad_itr), decltype(ldg_start)>));

    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE((is_same_v<ptr_type, int const*>));

    EXPECT_TRUE(ldg_start);
    EXPECT_NE(ldg_start, nullptr);
    EXPECT_NE(nullptr, ldg_start);
    EXPECT_EQ(std::accumulate(start, end, 0),
              std::accumulate(ldg_start, ldg_end, 0));
    EXPECT_EQ(static_cast<ptr_type>(ldg_start), some_data.data());

    EXPECT_TRUE((is_same_v<decltype(*ldg_start), RefInt>));
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
    EXPECT_TRUE((is_same_v<decltype(ctad_itr), decltype(ldg_start)>));
    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE((is_same_v<ptr_type, TestId const*>));
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
    EXPECT_TRUE((is_same_v<decltype(ctad_itr), decltype(ldg_start)>));
    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE((is_same_v<ptr_type, std::byte const*>));
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

}  // namespace test
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//
using LdgSpanTest = Test;

TEST_F(LdgSpanTest, pod)
{
    using LdgWrap = detail::LdgWrapper<int const>;
    using LdgIter = detail::LdgIterator<int const>;

    // LdgSpan must not be implicitly convertible to Span
    EXPECT_FALSE((std::is_constructible_v<Span<int const>, LdgSpan<int const>>))
        << "LdgSpan->Span conversion is prohibited";

    int const local_data[] = {123, 456, 789};
    Span<int const> basic_span(local_data);
    EXPECT_TRUE((is_same_v<decltype(basic_span[0]), int const&>));

    Span<LdgWrap> local_span(local_data);
    EXPECT_TRUE((is_same_v<typename Span<LdgWrap>::element_type, int const>));
    EXPECT_TRUE((is_same_v<decltype(local_span.data()), int const*>));
    EXPECT_TRUE((is_same_v<decltype(local_span.front()), LdgWrap>));
    EXPECT_TRUE((is_same_v<decltype(local_span.back()), LdgWrap>));
    EXPECT_TRUE((is_same_v<decltype(local_span[0]), LdgWrap>));
    EXPECT_TRUE((is_same_v<decltype(local_span.begin()), LdgIter>));
    EXPECT_TRUE((is_same_v<decltype(local_span.end()), LdgIter>));

    EXPECT_EQ(local_span.first(2).back(), 456);
    EXPECT_TRUE(
        (is_same_v<decltype(local_span), decltype(local_span.first(2))>));
    EXPECT_LT(local_span[0], local_span[1]);
    EXPECT_GT(local_span[2], local_span[1]);

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
    using LdgWrap = detail::LdgWrapper<TestId const>;
    using LdgIter = detail::LdgIterator<TestId const>;

    TestId local_data[] = {TestId{123}, TestId{456}, TestId{789}};
    Span<TestId> mutable_span(local_data);
    EXPECT_TRUE((is_same_v<decltype(mutable_span[0]), TestId&>));

    Span<LdgWrap> s(local_data);
    EXPECT_TRUE(
        (is_same_v<typename Span<LdgWrap>::element_type, TestId const>));
    EXPECT_TRUE((is_same_v<decltype(s.data()), TestId const*>));
    EXPECT_TRUE((is_same_v<decltype(s.front()), LdgWrap>));
    EXPECT_TRUE((is_same_v<decltype(s.back()), LdgWrap>));
    EXPECT_TRUE((is_same_v<decltype(s[0]), LdgWrap>));
    EXPECT_TRUE((is_same_v<decltype(s.begin()), LdgIter>));
    EXPECT_TRUE((is_same_v<decltype(s.end()), LdgIter>));

    EXPECT_EQ(s.first(2).back(), TestId{456});
    EXPECT_TRUE((is_same_v<decltype(s), decltype(s.first(2))>));
    EXPECT_EQ(s.subspan(1, 2)[1], TestId{789});
    EXPECT_LT(s[0], s[1]);
    EXPECT_GT(s[2], s[1]);

    auto begin = s.begin();
    EXPECT_EQ(*begin++, TestId{123});
    EXPECT_EQ(*begin++, TestId{456});
    EXPECT_EQ(*begin++, TestId{789});
    EXPECT_EQ(begin, s.end());
    EXPECT_EQ(s[2], TestId{789});
    EXPECT_EQ(s.end()[-3], TestId{123});
}

TEST_F(LdgSpanTest, quantity)
{
    static Dozen const eggs[] = {Dozen{1}, Dozen{3}, Dozen{5}};
    LdgSpan<Dozen const> view{eggs};
    ASSERT_EQ(3, view.size());
    EXPECT_EQ(dynamic_extent, view.extent);

    EXPECT_TRUE(
        (is_same_v<decltype(view.back()), detail::LdgWrapper<Dozen const>>));

    auto implicitly_converted = view.back() * 2;
    EXPECT_TRUE((is_same_v<decltype(implicitly_converted), Dozen>));
    EXPECT_EQ(Dozen{10}, implicitly_converted);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
