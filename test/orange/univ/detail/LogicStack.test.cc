//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/LogicStack.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/LogicStack.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(LogicStackTest, logic_stack)
{
    LogicStack s;
    EXPECT_EQ(0, s.size());
    EXPECT_TRUE(s.empty());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(s.pop(), DebugError);
        EXPECT_THROW(s.top(), DebugError);
        EXPECT_THROW(s.apply_not(), DebugError);
    }

    // Add a value [F]
    s.push(false);
    EXPECT_EQ(1, s.size());
    EXPECT_FALSE(s.top());

    for (bool val : {false, true, true, false, false, true, false})
    {
        s.push(val);
        EXPECT_EQ(val, s.top());
    }
    EXPECT_EQ(8, s.size());

    // Reverse order
    int ctr = s.size();
    for (bool val : {false, true, false, false, true, true, false, false})
    {
        EXPECT_EQ(val, s.pop()) << "Failed while popping element " << ctr;
        --ctr;
    }
    EXPECT_EQ(0, ctr);
    EXPECT_EQ(0, s.size());
    EXPECT_TRUE(s.empty());
}

TEST(LogicStackTest, operators)
{
    LogicStack s;
    s.push(false);
    EXPECT_FALSE(s.top());
    s.apply_not();
    EXPECT_TRUE(s.top());

    s.push(false);
    EXPECT_EQ(2, s.size());
    s.apply_or();
    EXPECT_EQ(true, s.top());
    EXPECT_EQ(1, s.size());

    s.push(true);
    s.apply_and();
    EXPECT_EQ(true, s.top());
    EXPECT_EQ(1, s.size());

    s.push(true);
    s.push(true);
    s.push(false);
    s.apply_and();
    EXPECT_FALSE(s.pop());
    EXPECT_TRUE(s.pop());
    EXPECT_EQ(1, s.size());

    s.push(false);
    s.push(false);
    s.apply_or();
    EXPECT_FALSE(s.top());
    EXPECT_EQ(2, s.size());

    s.apply_or();
    EXPECT_EQ(true, s.top());
    EXPECT_EQ(1, s.size());

    s.push(false);
    s.apply_and();
    EXPECT_FALSE(s.top());
    EXPECT_EQ(1, s.size());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
