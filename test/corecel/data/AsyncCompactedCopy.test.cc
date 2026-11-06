//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/AsyncCompactedCopy.hh"

#include "corecel/sys/Device.hh"

#include "AsyncCompactedCopy.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(AsyncCompactedCopyTest, unallocated)
{
    AsyncCompactedCopy<CompactTestItem> copier;
    EXPECT_FALSE(copier);
    EXPECT_EQ(0, copier.capacity());
    EXPECT_FALSE(copier.has_compacted());
    EXPECT_FALSE(copier.has_copied());
    EXPECT_TRUE(copier.take_copied().empty());

    copier = AsyncCompactedCopy<CompactTestItem>{};
    EXPECT_FALSE(copier);
}

TEST(AsyncCompactedCopyTest, TEST_IF_CELER_DEVICE(device))
{
    device().create_streams(1);

    VecVecInt batches{
        {1, 0, 2, 0, 3},
        {0, 0, 0, 0, 0},
        {4, 5, 0, 0, 6},
        {0, 0, 0, 0, 7},
    };
    auto result = compacted_copy_test(batches);

    ASSERT_EQ(batches.size(), result.size());
    EXPECT_VEC_EQ((std::vector<int>{1, 2, 3}), result[0]);
    EXPECT_TRUE(result[1].empty());
    EXPECT_VEC_EQ((std::vector<int>{4, 5, 6}), result[2]);
    EXPECT_VEC_EQ((std::vector<int>{7}), result[3]);
}

TEST(AsyncCompactedCopyTest, TEST_IF_CELER_DEVICE(device_assign))
{
    device().create_streams(1);

    auto result = assign_test({{1, 0, 2}, {0, 3, 0, 4, 5}, {6}});

    ASSERT_EQ(3, result.size());
    EXPECT_VEC_EQ((std::vector<int>{1, 2}), result[0]);
    EXPECT_VEC_EQ((std::vector<int>{3, 4, 5}), result[1]);
    EXPECT_VEC_EQ((std::vector<int>{6}), result[2]);
}

TEST(AsyncCompactedCopyTest, TEST_IF_CELER_DEVICE(device_assign_pending))
{
    device().create_streams(1);

    if constexpr (CELERITAS_DEBUG)
    {
        EXPECT_THROW(assign_pending_test(), DebugError);
    }
    else
    {
        EXPECT_NO_THROW(assign_pending_test());
    }
}

TEST(AsyncCompactedCopyTest, TEST_IF_CELER_DEVICE(device_destroy_pending))
{
    device().create_streams(1);

    EXPECT_NO_THROW(destroy_pending_test());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
