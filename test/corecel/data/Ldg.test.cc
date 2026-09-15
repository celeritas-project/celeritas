//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Ldg.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Ldg.hh"

#include <type_traits>

#include "corecel/OpaqueId.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
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
