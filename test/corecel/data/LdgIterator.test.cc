//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgIterator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/LdgIterator.hh"

#include <algorithm>
#include <iterator>
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

class LdgIteratorTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(LdgIteratorTest, integral_t)
{
    using VecInt = std::vector<int>;
    VecInt some_data = {1, 2, 3, 4};
    auto n = some_data.size();
    auto start = some_data.begin();
    auto end = some_data.end();
    auto ldg_start = make_LdgIterator(some_data.data());
    auto ldg_end = make_LdgIterator(some_data.data() + n);
    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE(ldg_start);
    EXPECT_EQ(std::accumulate(start, end, 0),
              std::accumulate(ldg_start, ldg_end, 0));
    EXPECT_EQ(static_cast<ptr_type>(ldg_start), some_data.data());
    EXPECT_EQ(*ldg_start++, 1);
    EXPECT_EQ(*ldg_start--, 2);
    EXPECT_EQ(*++ldg_start, 2);
    EXPECT_EQ(*--ldg_start, 1);
    EXPECT_EQ(ldg_start[n - 1], some_data.back());
    EXPECT_TRUE(ldg_end > ldg_start);
    auto ldg_start_copy = ldg_start;
    EXPECT_TRUE(ldg_start == ldg_start_copy);
    ldg_start += n;
    EXPECT_TRUE(ldg_start == ldg_end);
    ldg_end -= n;
    EXPECT_TRUE(ldg_end == ldg_start_copy);
    ldg_start.swap(ldg_end);
    EXPECT_TRUE(ldg_start == ldg_start_copy);
    EXPECT_TRUE(ldg_end == ldg_start + n);
    EXPECT_EQ(ldg_end - n, ldg_start);
    EXPECT_EQ(ldg_end - ldg_start, n);
    auto ldg_nullptr = LdgIterator<int>{nullptr};
    EXPECT_FALSE(ldg_nullptr);
}

TEST_F(LdgIteratorTest, opaqueid_t)
{
    using TestId = OpaqueId<struct Test>;
    using VecId = std::vector<typename TestId::size_type>;
    VecId some_data = {1, 2, 3, 4};
    auto n = some_data.size();
    auto ldg_start = LdgIterator<TestId>(some_data.data());
    auto ldg_end = LdgIterator<TestId>(some_data.data() + n);
    using ptr_type = typename decltype(ldg_start)::pointer;
    EXPECT_TRUE(ldg_start);
    EXPECT_EQ(static_cast<ptr_type>(ldg_start), some_data.data());
    EXPECT_EQ(*ldg_start++, TestId{1});
    EXPECT_EQ(*ldg_start--, TestId{2});
    EXPECT_EQ(*++ldg_start, TestId{2});
    EXPECT_EQ(*--ldg_start, TestId{1});
    EXPECT_EQ(ldg_start[n - 1], TestId{some_data.back()});
    EXPECT_TRUE(ldg_end > ldg_start);
    auto ldg_start_copy = ldg_start;
    EXPECT_TRUE(ldg_start == ldg_start_copy);
    ldg_start += n;
    EXPECT_TRUE(ldg_start == ldg_end);
    ldg_end -= n;
    EXPECT_TRUE(ldg_end == ldg_start_copy);
    ldg_start.swap(ldg_end);
    EXPECT_TRUE(ldg_start == ldg_start_copy);
    EXPECT_TRUE(ldg_end == ldg_start + n);
    EXPECT_EQ(ldg_end - n, ldg_start);
    EXPECT_EQ(ldg_end - ldg_start, n);
    auto ldg_nullptr = LdgIterator<int>{nullptr};
    EXPECT_FALSE(ldg_nullptr);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas