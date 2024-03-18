//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/CsgTreeUtils.hh"

#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/detail/InternalSurfaceFlagger.hh"
#include "orange/orangeinp/detail/PostfixLogicBuilder.hh"

#include "celeritas_test.hh"

using N = celeritas::orangeinp::NodeId;
using S = celeritas::LocalSurfaceId;
using celeritas::orangeinp::detail::InternalSurfaceFlagger;
using celeritas::orangeinp::detail::PostfixLogicBuilder;

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
std::string to_string(CsgTree const& tree)
{
    std::ostringstream os;
    os << tree;
    return os.str();
}

class CsgTreeUtilsTest : public ::celeritas::test::Test
{
  protected:
    template<class T>
    N insert(T&& n)
    {
        return tree_.insert(std::forward<T>(n)).first;
    }

  protected:
    CsgTree tree_;

    static constexpr auto true_id = CsgTree::true_node_id();
    static constexpr auto false_id = CsgTree::false_node_id();
};

constexpr NodeId CsgTreeUtilsTest::true_id;
constexpr NodeId CsgTreeUtilsTest::false_id;

TEST_F(CsgTreeUtilsTest, postfix_simplify)
{
    using LS = LocalSurfaceId;

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
    auto bdy_outer = this->insert(S{4});
    auto bdy = this->insert(Joined{op_and, {bdy_outer, mz, below_pz}});
    auto zslab = this->insert(Joined{op_and, {mz, below_pz}});

    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: "
        "surface 2, 6: not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: "
        "all{2,4,9}, 11: not{7}, 12: all{10,11}, 13: surface 4, 14: "
        "all{2,4,13}, 15: all{2,4}, }",
        to_string(tree_));

    // Test postfix and internal surface flagger
    InternalSurfaceFlagger has_internal_surfaces(tree_);
    PostfixLogicBuilder build_postfix(tree_);
    {
        EXPECT_FALSE(has_internal_surfaces(mz));
        auto&& [faces, lgc] = build_postfix(mz);

        static size_type expected_lgc[] = {0};
        static LS const expected_faces[] = {LS{0u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
    {
        EXPECT_FALSE(has_internal_surfaces(below_pz));
        auto&& [faces, lgc] = build_postfix(below_pz);

        static size_type expected_lgc[] = {0, logic::lnot};
        static LS const expected_faces[] = {LS{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
    {
        EXPECT_FALSE(has_internal_surfaces(zslab));
        auto&& [faces, lgc] = build_postfix(zslab);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land};
        static LS const expected_faces[] = {LS{0u}, LS{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
    {
        EXPECT_FALSE(has_internal_surfaces(zslab));
        auto&& [faces, lgc] = build_postfix(inner_cyl);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land, 2u, logic::lnot, logic::land};
        static LS const expected_faces[] = {LS{0u}, LS{1u}, LS{2u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ("all(+0, -1, -2)", build_infix_string(tree_, inner_cyl));
    }
    {
        EXPECT_TRUE(has_internal_surfaces(shell));
        auto&& [faces, lgc] = build_postfix(shell);

        static size_type const expected_lgc[] = {
            0u,
            1u,
            logic::lnot,
            logic::land,
            3u,
            logic::lnot,
            logic::land,
            0u,
            1u,
            logic::lnot,
            logic::land,
            2u,
            logic::lnot,
            logic::land,
            logic::lnot,
            logic::land,
        };
        static LS const expected_faces[] = {LS{0u}, LS{1u}, LS{2u}, LS{3u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ("all(all(+0, -1, -3), !all(+0, -1, -2))",
                  build_infix_string(tree_, shell));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(bdy));
        auto&& [faces, lgc] = build_postfix(bdy);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land, 2u, logic::land};
        static LS const expected_faces[] = {LS{0u}, LS{1u}, LS{4u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
        EXPECT_EQ("all(+0, -1, +4)", build_infix_string(tree_, bdy));
    }

    // Imply inside boundary
    auto min_node = replace_down(&tree_, bdy, True{});
    EXPECT_EQ(mz, min_node);

    // Save tree for later
    CsgTree unsimplified_tree{tree_};

    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{0}, 3: ->{1}, 4: ->{0}, 5: surface 2, 6: "
        "not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: all{2,4,9}, 11: "
        "not{7}, 12: all{10,11}, 13: ->{0}, 14: ->{0}, 15: all{2,4}, }",
        to_string(tree_));

    // Simplify once: first simplification is the inner cylinder
    min_node = simplify_up(&tree_, min_node);
    EXPECT_EQ(NodeId{7}, min_node);
    EXPECT_EQ("all(-3, !-2)", build_infix_string(tree_, shell));

    // Simplify again: the shell is simplified this time
    min_node = simplify_up(&tree_, min_node);
    EXPECT_EQ(NodeId{11}, min_node);
    EXPECT_EQ("all(+2, -3)", build_infix_string(tree_, shell));

    // Simplify one final time: nothing further is simplified
    min_node = simplify_up(&tree_, min_node);
    EXPECT_EQ(NodeId{}, min_node);
    EXPECT_EQ("all(+2, -3)", build_infix_string(tree_, shell));

    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{0}, 3: ->{1}, 4: ->{0}, 5: surface 2, 6: "
        "not{5}, 7: ->{6}, 8: surface 3, 9: not{8}, 10: ->{9}, 11: ->{5}, 12: "
        "all{5,9}, 13: ->{0}, 14: ->{0}, 15: ->{0}, }",
        to_string(tree_));

    // Try simplifying recursively from the original minimum node
    simplify(&unsimplified_tree, mz);
    EXPECT_EQ(to_string(tree_), to_string(unsimplified_tree));

    // Test postfix builder with remapping
    {
        auto remapped_surf = calc_surfaces(tree_);
        static LS const expected_remapped_surf[] = {LS{2u}, LS{3u}};
        EXPECT_VEC_EQ(expected_remapped_surf, remapped_surf);

        PostfixLogicBuilder build_postfix(tree_, remapped_surf);
        auto&& [faces, lgc] = build_postfix(shell);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land};
        static LS const expected_faces[] = {LS{0u}, LS{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
}

TEST_F(CsgTreeUtilsTest, replace_union)
{
    auto a = this->insert(S{0});
    auto b = this->insert(S{1});
    auto inside_a = this->insert(Negated{a});
    auto inside_b = this->insert(Negated{b});
    auto inside_a_or_b = this->insert(Joined{op_or, {inside_a, inside_b}});

    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: "
        "not{3}, 6: any{4,5}, }",
        to_string(tree_));

    // Imply inside neither
    auto min_node = replace_down(&tree_, inside_a_or_b, False{});
    EXPECT_EQ(a, min_node);
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{0}, 3: ->{0}, 4: ->{1}, 5: ->{1}, 6: "
        "->{1}, }",
        to_string(tree_));

    min_node = simplify_up(&tree_, min_node);
    EXPECT_EQ(NodeId{}, min_node);
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{0}, 3: ->{0}, 4: ->{1}, 5: ->{1}, 6: "
        "->{1}, }",
        to_string(tree_));
}

TEST_F(CsgTreeUtilsTest, replace_union_2)
{
    auto a = this->insert(S{0});
    auto b = this->insert(S{1});
    auto inside_a = this->insert(Negated{a});
    this->insert(Negated{b});
    auto outside_a_or_b = this->insert(Joined{op_or, {a, b}});
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: "
        "not{3}, 6: any{2,3}, }",
        to_string(tree_));

    // Imply !(a | b) -> a & b
    auto min_node = replace_down(&tree_, outside_a_or_b, False{});
    EXPECT_EQ(a, min_node);
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{1}, 3: ->{1}, 4: not{2}, 5: not{3}, 6: "
        "->{1}, }",
        to_string(tree_));

    min_node = simplify_up(&tree_, min_node);
    EXPECT_EQ(inside_a, min_node);

    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{1}, 3: ->{1}, 4: ->{0}, 5: ->{0}, 6: "
        "->{1}, }",
        to_string(tree_));

    // No simplification
    min_node = simplify_up(&tree_, min_node);
    EXPECT_EQ(NodeId{}, min_node);
}

TEST_F(CsgTreeUtilsTest, calc_surfaces)
{
    this->insert(S{3});
    auto s1 = this->insert(S{1});
    this->insert(Negated{s1});
    this->insert(S{1});

    EXPECT_EQ((std::vector<S>{S{1}, S{3}}), calc_surfaces(tree_));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
