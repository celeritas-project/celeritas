//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/CsgTree.test.cc
//---------------------------------------------------------------------------//
#include "orange/construct/CsgTree.hh"

#include "celeritas_config.h"

#include "celeritas_test.hh"

#if CELERITAS_USE_JSON
#    include "orange/construct/CsgTreeIO.json.hh"
#endif

using N = celeritas::csg::NodeId;
using S = celeritas::LocalSurfaceId;

namespace celeritas
{
namespace csg
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(CsgTypes, hash)
{
    std::hash<Node> variant_hash;
    EXPECT_NE(variant_hash(True{}), variant_hash(False{}));
    EXPECT_NE(variant_hash(Aliased{N{0}}), variant_hash(Surface{S{0}}));
}

//---------------------------------------------------------------------------//
TEST(CsgTypes, stream)
{
    auto to_string = [](auto&& n) {
        std::ostringstream os;
        os << n;
        return os.str();
    };

    // Raw types
    EXPECT_EQ("true", to_string(True{}));
    EXPECT_EQ("false", to_string(False{}));
    EXPECT_EQ("->{123}", to_string(Aliased{N{123}}));
    EXPECT_EQ("not{456}", to_string(Negated{N{456}}));
    EXPECT_EQ("surface 4", to_string(Surface{S{4}}));
    EXPECT_EQ("all{1,2,4}", to_string(Joined{op_and, {N{1}, N{2}, N{4}}}));
    EXPECT_EQ("any{0,1}", to_string(Joined{op_or, {N{0}, N{1}}}));

    // With visitor
    EXPECT_EQ("not{123}", std::visit(to_string, Node{Negated{N{123}}}));

    // With wrapper
    EXPECT_EQ("not{123}", to_string(Node{Negated{N{123}}}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace csg

namespace test
{
//---------------------------------------------------------------------------//
using namespace celeritas::csg;

class CsgTreeTest : public ::celeritas::test::Test
{
  protected:
    CsgTree tree_;

    static constexpr auto true_id = CsgTree::true_node_id();
    static constexpr auto false_id = CsgTree::false_node_id();

    template<class T>
    N insert(T&& n)
    {
        return tree_.insert(std::forward<T>(n));
    }

    std::string to_json_string() const
    {
#if CELERITAS_USE_JSON
        nlohmann::json obj{tree_};
        return obj.dump();
#else
        return {};
#endif
    }
};

constexpr NodeId CsgTreeTest::true_id;
constexpr NodeId CsgTreeTest::false_id;

TEST_F(CsgTreeTest, true_false)
{
    EXPECT_EQ(2, tree_.size());  // True and false added by default
    EXPECT_EQ(true_id, this->insert(True{}));
    EXPECT_EQ(true_id, this->insert(Negated{false_id}));
    EXPECT_EQ(false_id, this->insert(False{}));
    EXPECT_EQ(false_id, this->insert(Negated{true_id}));

    EXPECT_EQ(Node{True{}}, tree_[true_id]);
    EXPECT_EQ(Node{Negated{true_id}}, tree_[false_id]);
}

TEST_F(CsgTreeTest, TEST_IF_CELERITAS_DEBUG(prohibited_insertion))
{
    // Try prohibited cases
    EXPECT_THROW(this->insert(Negated{N{5}}), DebugError);
    EXPECT_THROW(this->insert(S{}), DebugError);
    EXPECT_THROW(this->insert(Joined{op_and, {N{3}}}), DebugError);
    EXPECT_THROW(this->insert(Joined{OperatorToken::lnot, {N{0}}}), DebugError);
}

TEST_F(CsgTreeTest, surfaces)
{
    // Test deduplication and add two surfaces
    EXPECT_EQ(N{2}, this->insert(S{1}));
    EXPECT_EQ(Node{Surface{S{1}}}, tree_[N{2}]);
    EXPECT_EQ(N{3}, this->insert(S{3}));
    EXPECT_EQ(N{2}, this->insert(S{1}));
}

TEST_F(CsgTreeTest, negation)
{
    EXPECT_EQ(N{2}, this->insert(S{1}));
    EXPECT_EQ(N{3}, this->insert(Negated{N{2}}));
    EXPECT_EQ(N{2}, this->insert(Negated{N{3}}));
}

TEST_F(CsgTreeTest, join)
{
    EXPECT_EQ(N{2}, this->insert(S{1}));
    EXPECT_EQ(N{3}, this->insert(S{3}));

    // Sort and deduplicate
    EXPECT_EQ(N{4}, this->insert(Joined{op_and, {N{3}, N{2}, N{3}}}));
    auto actual = tree_[N{4}];
    ASSERT_TRUE(std::holds_alternative<Joined>(actual));
    auto const& j = std::get<Joined>(actual);
    EXPECT_EQ(op_and, j.op);
    ASSERT_EQ(2, j.nodes.size());
    EXPECT_EQ(N{2}, j.nodes[0]);
    EXPECT_EQ(N{3}, j.nodes[1]);

    // Single-node case
    EXPECT_EQ(N{2}, this->insert(Joined{op_and, {N{2}}}));
    EXPECT_EQ(true_id, this->insert(Joined{op_or, {true_id}}));

    // Empty cases
    EXPECT_EQ(true_id, this->insert(Joined{op_and, {}}));
    EXPECT_EQ(false_id, this->insert(Joined{op_or, {}}));

    // True/false autosimplification
    EXPECT_EQ(true_id, this->insert(Joined{op_or, {N{2}, true_id}}));
    EXPECT_EQ(false_id, this->insert(Joined{op_and, {N{3}, false_id, N{2}}}));
}

TEST_F(CsgTreeTest, manual_simplify)
{
    auto mz = this->insert(S{0});
    auto pz = this->insert(S{1});
    auto below_pz = this->insert(Negated{pz});
    auto r_inner = this->insert(S{2});
    auto inside_inner = this->insert(Negated{r_inner});
    auto inner_cyl = this->insert(Joined{op_and, {mz, below_pz, inside_inner}});
    auto r_outer = this->insert(S{3});
    auto inside_outer = this->insert(Negated{r_outer});
    auto outer_cyl = this->insert(Joined{op_and, {mz, below_pz, inside_outer}});
    auto not_inner = this->insert(Negated{inner_cyl});
    auto shell = this->insert(Joined{op_and, {not_inner, outer_cyl}});
    auto sphere = this->insert(S{4});
    auto below_mz = this->insert(Negated{mz});
    auto dumb_union = this->insert(Joined{op_or, {sphere, below_mz}});

    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(
            R"json([["t",["~",0],["S",0],["S",1],["~",3],["S",2],["~",5],["&",[2,4,6]],["S",3],["~",8],["&",[2,4,9]],["~",7],["&",[10,11]],["S",4],["~",2],["|",[13,14]]]])json",
            this->to_json_string());
    }

    // Suppose we implied above mz and below pz: sweep down
    {
        // implied below_pz true
        auto old = tree_.exchange(below_pz, True{});
        EXPECT_EQ(Node{Negated{pz}}, old);
        EXPECT_EQ(Node{Aliased{true_id}}, tree_[below_pz]);
    }
    {
        // implies above pz is false
        auto old = tree_.exchange(pz, False{});
        EXPECT_EQ(Node{Surface{S{1}}}, old);
        EXPECT_EQ(Node{Aliased{false_id}}, tree_[pz]);
    }
    {
        // above mz is true
        auto old = tree_.exchange(mz, True{});
        EXPECT_EQ(Node{Surface{S{0}}}, old);
        EXPECT_EQ(Node{Aliased{true_id}}, tree_[mz]);
    }

    // Sweep up
    {
        // Inner cyl simplifies to inside_inner
        tree_.exchange(inner_cyl, Joined{op_and, {inside_inner}});
        EXPECT_EQ(Node{Aliased{inside_inner}}, tree_[inner_cyl]);
    }
    {
        // Outer cyl simplifies to r_outer
        tree_.exchange(outer_cyl, Joined{op_and, {inside_outer}});
        EXPECT_EQ(Node{Aliased{inside_outer}}, tree_[outer_cyl]);
    }
    {
        // Negated inner simplifies to r_inner
        tree_.exchange(not_inner, Negated{inside_inner});
        EXPECT_EQ(Node{Aliased{r_inner}}, tree_[not_inner]);
    }
    {
        // Shell combines join
        tree_.exchange(shell, Joined{op_and, {not_inner, inside_outer}});
        EXPECT_EQ("all{5,9}", to_string(tree_[shell]));
    }
    {
        // below mz is false
        tree_.exchange(below_mz, False{});
        EXPECT_EQ(Node{Aliased{false_id}}, tree_[below_mz]);
    }
    {
        // Dumb union simplifies to "sphere"
        tree_.exchange(dumb_union, Joined{op_or, {sphere}});
        EXPECT_EQ(Node{Aliased{sphere}}, tree_[dumb_union]);
    }

    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(
            R"json([["t",["~",0],["=",0],["=",1],["=",0],["S",2],["~",5],["=",6],["S",3],["~",8],["=",9],["=",5],["&",[5,9]],["S",4],["=",1],["=",13]]])json",
            this->to_json_string());
    }
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
