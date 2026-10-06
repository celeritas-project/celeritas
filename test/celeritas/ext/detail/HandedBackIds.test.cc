//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HandedBackIds.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/HandedBackIds.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(HandedBackIdsTest, same_event)
{
    HandedBackIds ids;
    EXPECT_EQ(0, ids.size());
    EXPECT_FALSE(ids.erase(0, 5));

    ids.insert(0, 5);
    ids.insert(0, 7);
    EXPECT_EQ(2, ids.size());

    // A track not handed back (e.g., a new track allocated at the address of
    // a deleted handed-back track) has a different ID
    EXPECT_FALSE(ids.erase(0, 6));

    // Each handed-back track is processed once
    EXPECT_TRUE(ids.erase(0, 5));
    EXPECT_FALSE(ids.erase(0, 5));
    EXPECT_EQ(1, ids.size());

    // A suspended track can be handed back again with the same ID
    ids.insert(0, 5);
    EXPECT_TRUE(ids.erase(0, 5));
}

TEST(HandedBackIdsTest, new_event)
{
    HandedBackIds ids;
    ids.insert(0, 5);
    ids.insert(0, 7);

    // Track IDs restart in a new event: leftover IDs are discarded
    EXPECT_FALSE(ids.erase(1, 5));
    EXPECT_EQ(0, ids.size());
    EXPECT_FALSE(ids.erase(1, 7));

    ids.insert(1, 7);
    EXPECT_EQ(1, ids.size());
    ids.insert(2, 3);
    EXPECT_EQ(1, ids.size());
    EXPECT_FALSE(ids.erase(2, 7));
    EXPECT_TRUE(ids.erase(2, 3));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
