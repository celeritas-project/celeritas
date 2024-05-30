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
#include "orange/orangeinp/detail/SenseEvaluator.hh"
#include "orange/surf/VariantSurface.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

using N = celeritas::orangeinp::NodeId;
using S = celeritas::LocalSurfaceId;
using celeritas::orangeinp::detail::InternalSurfaceFlagger;
using celeritas::orangeinp::detail::PostfixLogicBuilder;

namespace celeritas
{
std::ostream& operator<<(std::ostream& os, SignedSense s)
{
    return (os << to_cstring(s));
}

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

    template<class T>
    N insert_surface(T&& surf)
    {
        LocalSurfaceId lsid{static_cast<size_type>(surfaces_.size())};
        surfaces_.push_back(std::forward<T>(surf));
        return this->insert(lsid);
    }

    SignedSense is_inside(NodeId n, Real3 const& point) const
    {
        CELER_EXPECT(n < tree_.size());
        detail::SenseEvaluator eval_sense(tree_, surfaces_, point);
        return eval_sense(n);
    }

  protected:
    CsgTree tree_;
    std::vector<VariantSurface> surfaces_;

    static constexpr auto true_id = CsgTree::true_node_id();
    static constexpr auto false_id = CsgTree::false_node_id();
};

constexpr NodeId CsgTreeUtilsTest::true_id;
constexpr NodeId CsgTreeUtilsTest::false_id;

TEST_F(CsgTreeUtilsTest, postfix_simplify)
{
    // NOTE: mz = below = "false"
    auto mz = this->insert_surface(PlaneZ{-1.0});
    auto pz = this->insert_surface(PlaneZ{1.0});
    auto below_pz = this->insert(Negated{pz});
    auto r_inner = this->insert_surface(CCylZ{0.5});
    auto inside_inner = this->insert(Negated{r_inner});
    auto inner_cyl = this->insert(Joined{op_and, {mz, below_pz, inside_inner}});
    auto r_outer = this->insert_surface(CCylZ{1.0});
    auto inside_outer = this->insert(Negated{r_outer});
    auto outer_cyl = this->insert(Joined{op_and, {mz, below_pz, inside_outer}});
    auto not_inner = this->insert(Negated{inner_cyl});
    auto shell = this->insert(Joined{op_and, {not_inner, outer_cyl}});
    auto bdy_outer = this->insert_surface(CCylZ{4.0});
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
        static S const expected_faces[] = {S{0u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        // NOTE: inside and outside are flipped
        static_assert(Sense::inside == to_sense(false));
        EXPECT_EQ(SignedSense::outside, is_inside(mz, {0, 0, -2}));
        EXPECT_EQ(SignedSense::on, is_inside(mz, {0, 0, -1}));
        EXPECT_EQ(SignedSense::inside, is_inside(mz, {0, 0, 2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(below_pz));
        auto&& [faces, lgc] = build_postfix(below_pz);

        static size_type expected_lgc[] = {0, logic::lnot};
        static S const expected_faces[] = {S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ(SignedSense::inside, is_inside(below_pz, {0, 0, 0.5}));
        EXPECT_EQ(SignedSense::on, is_inside(below_pz, {0, 0, 1}));
        EXPECT_EQ(SignedSense::outside, is_inside(below_pz, {0, 0, 2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(zslab));
        auto&& [faces, lgc] = build_postfix(zslab);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land};
        static S const expected_faces[] = {S{0u}, S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ(SignedSense::inside, is_inside(zslab, {0, 0, 0}));
        EXPECT_EQ(SignedSense::on, is_inside(zslab, {0, 0, 1}));
        EXPECT_EQ(SignedSense::outside, is_inside(zslab, {0, 0, -2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(inner_cyl));
        auto&& [faces, lgc] = build_postfix(inner_cyl);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land, 2u, logic::lnot, logic::land};
        static S const expected_faces[] = {S{0u}, S{1u}, S{2u}};
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
        static S const expected_faces[] = {S{0u}, S{1u}, S{2u}, S{3u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ("all(all(+0, -1, -3), !all(+0, -1, -2))",
                  build_infix_string(tree_, shell));

        EXPECT_EQ(SignedSense::outside, is_inside(shell, {0, 0, 0}));
        EXPECT_EQ(SignedSense::on, is_inside(shell, {0, 0, 1}));
        EXPECT_EQ(SignedSense::inside, is_inside(shell, {0.75, 0, 0}));
        EXPECT_EQ(SignedSense::outside, is_inside(shell, {1.25, 0, 0}));
        EXPECT_EQ(SignedSense::outside, is_inside(shell, {0, 0, -2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(bdy));
        auto&& [faces, lgc] = build_postfix(bdy);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land, 2u, logic::land};
        static S const expected_faces[] = {S{0u}, S{1u}, S{4u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
        EXPECT_EQ("all(+0, -1, +4)", build_infix_string(tree_, bdy));
    }

    // Imply inside boundary
    replace_and_simplify(&tree_, bdy, True{});
#if 0
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
        static S const expected_remapped_surf[] = {S{2u}, S{3u}};
        EXPECT_VEC_EQ(expected_remapped_surf, remapped_surf);

        PostfixLogicBuilder build_postfix(tree_, remapped_surf);
        auto&& [faces, lgc] = build_postfix(shell);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land};
        static S const expected_faces[] = {S{0u}, S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
#endif
}

TEST_F(CsgTreeUtilsTest, tilecal_bug)
{
    EXPECT_EQ(N{2}, this->insert(Surface{S{0}}));  // mz
    EXPECT_EQ(N{3}, this->insert(Surface{S{1}}));  // pz
    EXPECT_EQ(N{4}, this->insert(Negated{N{3}}));
    EXPECT_EQ(N{5}, this->insert(Surface{S{2}}));  // interior.cz
    EXPECT_EQ(N{6}, this->insert(Negated{N{5}}));
    EXPECT_EQ(N{7},
              this->insert(
                  Joined{op_and, {N{2}, N{4}, N{6}}}));  // TileTBEnv.interior
    EXPECT_EQ(N{8}, this->insert(Surface{S{3}}));  // excluded.cz
    EXPECT_EQ(N{9}, this->insert(Negated{N{8}}));
    EXPECT_EQ(N{10},
              this->insert(
                  Joined{op_and, {N{2}, N{4}, N{9}}}));  // TileTBEnv.excluded
    EXPECT_EQ(N{11}, this->insert(Negated{N{10}}));
    EXPECT_EQ(N{12}, this->insert(Surface{S{4}}));
    EXPECT_EQ(N{13}, this->insert(Surface{S{5}}));
    EXPECT_EQ(N{14},
              this->insert(Joined{op_and, {N{12}, N{13}}}));  // TileTBEnv.angle
    EXPECT_EQ(N{15},
              this->insert(Joined{op_and, {N{7}, N{11}, N{14}}}));  // TileTBEnv
    EXPECT_EQ(N{16}, this->insert(Negated{N{15}}));  // [EXTERIOR]
    EXPECT_EQ(N{17}, this->insert(Surface{S{6}}));  // Barrel.angle.p0
    EXPECT_EQ(N{18}, this->insert(Surface{S{7}}));  // Barrel.angle.p1
    EXPECT_EQ(N{19}, this->insert(Negated{N{18}}));
    EXPECT_EQ(
        N{20},
        this->insert(Joined{op_and, {N{6}, N{17}, N{19}}}));  // Barrel.interior
    EXPECT_EQ(
        N{21},
        this->insert(Joined{op_and, {N{9}, N{17}, N{19}}}));  // Barrel.excluded
    EXPECT_EQ(N{22}, this->insert(Negated{N{21}}));
    EXPECT_EQ(N{23}, this->insert(Surface{S{8}}));  // Barrel.angle.p0
    EXPECT_EQ(N{24}, this->insert(Surface{S{9}}));  // Barrel.angle.p1
    EXPECT_EQ(N{25},
              this->insert(Joined{op_and, {N{23}, N{24}}}));  // Barrel.angle
    EXPECT_EQ(N{26},
              this->insert(Joined{op_and, {N{20}, N{22}, N{25}}}));  // Barrel
    EXPECT_EQ(N{27}, this->insert(Negated{N{26}}));
    EXPECT_EQ(N{28}, this->insert(Joined{op_and, {N{15}, N{27}}}));

    EXPECT_EQ(29, tree_.size());

    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: "
        "surface 2, 6: not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: "
        "all{2,4,9}, 11: not{10}, 12: surface 4, 13: surface 5, 14: "
        "all{12,13}, 15: all{2,4,6,11,12,13}, 16: not{15}, 17: surface 6, 18: "
        "surface 7, 19: not{18}, 20: all{6,17,19}, 21: all{9,17,19}, 22: "
        "not{21}, 23: surface 8, 24: surface 9, 25: all{23,24}, 26: "
        "all{6,17,19,22,23,24}, 27: not{26}, 28: all{2,4,6,11,12,13,27}, }",
        to_string(tree_));

    EXPECT_EQ("!all(+0, -1, -2, !all(+0, -1, -3), +4, +5)",
              build_infix_string(tree_, N{16}));
    replace_and_simplify(&tree_, N{16}, False{});
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{0}, 3: ->{1}, 4: ->{0}, 5: ->{1}, 6: "
        "->{0}, 7: ->{0}, 8: ->{0}, 9: ->{1}, 10: ->{1}, 11: ->{0}, 12: "
        "->{0}, 13: ->{0}, 14: ->{0}, 15: ->{0}, 16: ->{1}, 17: surface 6, "
        "18: surface 7, 19: not{18}, 20: all{17,19}, 21: ->{1}, 22: ->{0}, "
        "23: surface 8, 24: surface 9, 25: all{23,24}, 26: all{17,19,23,24}, "
        "27: not{26}, 28: ->{27}, }",
        to_string(tree_));
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
    replace_and_simplify(&tree_, inside_a_or_b, False{});
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{0}, 3: ->{0}, 4: ->{1}, 5: ->{1}, 6: "
        "->{1}, }",
        to_string(tree_));
}

TEST_F(CsgTreeUtilsTest, replace_union_2)
{
    auto a = this->insert(S{0});
    auto b = this->insert(S{1});
    this->insert(Negated{b});
    auto outside_a_or_b = this->insert(Joined{op_or, {a, b}});
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: "
        "not{3}, 6: any{2,3}, }",
        to_string(tree_));

    // Imply !(a | b) -> a & b
    replace_and_simplify(&tree_, outside_a_or_b, False{});
    EXPECT_EQ(
        "{0: true, 1: not{0}, 2: ->{1}, 3: ->{1}, 4: ->{0}, 5: ->{0}, 6: "
        "->{1}, }",
        to_string(tree_));
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
