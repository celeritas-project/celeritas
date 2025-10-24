//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgIterator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/LdgIterator.hh"

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include "corecel/OpaqueId.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using LdgIteratorTest = Test;

TEST_F(LdgIteratorTest, arithmetic_t)
{
    using VecInt = std::vector<int>;
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
}  // namespace test
}  // namespace celeritas
