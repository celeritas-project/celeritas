//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/CsgTreeUtils.hh"

#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTypes.hh"
#include "orange/orangeinp/detail/CsgLogicUtils.hh"
#include "orange/orangeinp/detail/InternalSurfaceFlagger.hh"
#include "orange/orangeinp/detail/SenseEvaluator.hh"
#include "orange/surf/VariantSurface.hh"

#include "CsgTestUtils.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

using N = celeritas::orangeinp::NodeId;
using S = celeritas::LocalSurfaceId;
using celeritas::orangeinp::detail::build_logic;
using celeritas::orangeinp::detail::InfixBuildLogicPolicy;
using celeritas::orangeinp::detail::InternalSurfaceFlagger;
using celeritas::orangeinp::detail::PostfixBuildLogicPolicy;

namespace celeritas
{
std::ostream& operator<<(std::ostream& os, SignedSense s)
{
    return (os << to_cstring(s));
}

template<>
struct ReprTraits<N>
{
    static void print_type(std::ostream&, char const* = nullptr) {}
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, N value)
    {
        orangeinp::test::stream_node_id(os, value);
    }
};

template<>
struct ReprTraits<S>
{
    static void print_type(std::ostream&, char const* = nullptr) {}
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, S s)
    {
        os << "S{";
        if (s)
        {
            os << s.unchecked_get();
        }
        os << '}';
    }
};

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

    auto always_false = this->insert(Joined{op_and, {shell, inner_cyl}});

    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: surface 2, 6: not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: all{2,4,9}, 11: not{7}, 12: all{2,4,9,11}, 13: surface 4, 14: all{2,4,13}, 15: all{2,4}, 16: all{2,4,6,9,11}, })",
        to_string(tree_));

    // Test postfix and internal surface flagger
    InternalSurfaceFlagger has_internal_surfaces(tree_);
    auto build_postfix =
        [&](N n, detail::BuildLogicResult::VecSurface* mapping = nullptr) {
            if (mapping)
            {
                return build_logic(PostfixBuildLogicPolicy{tree_, *mapping}, n);
            }
            return build_logic(PostfixBuildLogicPolicy{tree_}, n);
        };

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

        EXPECT_EQ("all(+0, -1, -3, !all(+0, -1, -2))",
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
    {
        EXPECT_TRUE(has_internal_surfaces(always_false));
        auto&& [faces, lgc] = build_postfix(always_false);

        static size_type const expected_lgc[] = {
            0u,          1u,          logic::lnot, logic::land, 2u,
            logic::lnot, logic::land, 3u,          logic::lnot, logic::land,
            0u,          1u,          logic::lnot, logic::land, 2u,
            logic::lnot, logic::land, logic::lnot, logic::land,
        };
        static S const expected_faces[] = {S{0}, S{1}, S{2}, S{3}};
        EXPECT_VEC_EQ(expected_lgc, lgc) << ReprLogic{lgc};
        EXPECT_VEC_EQ(expected_faces, faces);
        EXPECT_EQ("all(+0, -1, -2, -3, !all(+0, -1, -2))",
                  build_infix_string(tree_, always_false));
    }

    // Imply inside boundary
    replace_and_simplify(&tree_, bdy, True{});

    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: ->{0}, 3: ->{1}, 4: ->{0}, 5: surface 2, 6: not{5}, 7: ->{6}, 8: surface 3, 9: not{8}, 10: ->{9}, 11: ->{5}, 12: all{5,9}, 13: ->{0}, 14: ->{0}, 15: ->{0}, 16: all{5,6,9}, })",
        to_string(tree_));

    // Test postfix builder with remapping
    {
        auto remapped_surf = calc_surfaces(tree_);
        static S const expected_remapped_surf[] = {S{2u}, S{3u}};
        EXPECT_VEC_EQ(expected_remapped_surf, remapped_surf);

        auto&& [faces, lgc] = build_postfix(shell, &remapped_surf);

        static size_type const expected_lgc[]
            = {0u, 1u, logic::lnot, logic::land};
        static S const expected_faces[] = {S{0u}, S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
}

TEST_F(CsgTreeUtilsTest, infix_simplify)
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
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: surface 2, 6: not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: all{2,4,9}, 11: not{7}, 12: all{2,4,9,11}, 13: surface 4, 14: all{2,4,13}, 15: all{2,4}, })",
        to_string(tree_));

    // Test infix and internal surface flagger
    InternalSurfaceFlagger has_internal_surfaces(tree_);
    auto build_infix =
        [&](N n, detail::BuildLogicResult::VecSurface* mapping = nullptr) {
            if (mapping)
            {
                return build_logic(InfixBuildLogicPolicy{tree_, *mapping}, n);
            }
            return build_logic(InfixBuildLogicPolicy{tree_}, n);
        };
    {
        EXPECT_FALSE(has_internal_surfaces(mz));
        auto&& [faces, lgc] = build_infix(mz);

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
        auto&& [faces, lgc] = build_infix(below_pz);

        static size_type expected_lgc[] = {logic::lnot, 0};
        static S const expected_faces[] = {S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ(SignedSense::inside, is_inside(below_pz, {0, 0, 0.5}));
        EXPECT_EQ(SignedSense::on, is_inside(below_pz, {0, 0, 1}));
        EXPECT_EQ(SignedSense::outside, is_inside(below_pz, {0, 0, 2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(zslab));
        auto&& [faces, lgc] = build_infix(zslab);

        static size_type const expected_lgc[]
            = {logic::lopen, 0u, logic::land, logic::lnot, 1u, logic::lclose};
        static S const expected_faces[] = {S{0u}, S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ(SignedSense::inside, is_inside(zslab, {0, 0, 0}));
        EXPECT_EQ(SignedSense::on, is_inside(zslab, {0, 0, 1}));
        EXPECT_EQ(SignedSense::outside, is_inside(zslab, {0, 0, -2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(inner_cyl));
        auto&& [faces, lgc] = build_infix(inner_cyl);

        static size_type const expected_lgc[] = {logic::lopen,
                                                 0u,
                                                 logic::land,
                                                 logic::lnot,
                                                 1u,
                                                 logic::land,
                                                 logic::lnot,
                                                 2u,
                                                 logic::lclose};
        static S const expected_faces[] = {S{0u}, S{1u}, S{2u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ("all(+0, -1, -2)", build_infix_string(tree_, inner_cyl));
    }
    {
        EXPECT_TRUE(has_internal_surfaces(shell));
        auto&& [faces, lgc] = build_infix(shell);

        static size_type const expected_lgc[] = {
            logic::lopen,
            0u,
            logic::land,
            logic::lnot,
            1u,
            logic::land,
            logic::lnot,
            3u,
            logic::land,
            logic::lnot,
            logic::lopen,
            0u,
            logic::land,
            logic::lnot,
            1u,
            logic::land,
            logic::lnot,
            2u,
            logic::lclose,
            logic::lclose,
        };
        static S const expected_faces[] = {S{0u}, S{1u}, S{2u}, S{3u}};
        EXPECT_VEC_EQ(expected_lgc, lgc) << ReprLogic{lgc};
        EXPECT_VEC_EQ(expected_faces, faces);

        EXPECT_EQ("all(+0, -1, -3, !all(+0, -1, -2))",
                  build_infix_string(tree_, shell));

        EXPECT_EQ(SignedSense::outside, is_inside(shell, {0, 0, 0}));
        EXPECT_EQ(SignedSense::on, is_inside(shell, {0, 0, 1}));
        EXPECT_EQ(SignedSense::inside, is_inside(shell, {0.75, 0, 0}));
        EXPECT_EQ(SignedSense::outside, is_inside(shell, {1.25, 0, 0}));
        EXPECT_EQ(SignedSense::outside, is_inside(shell, {0, 0, -2}));
    }
    {
        EXPECT_FALSE(has_internal_surfaces(bdy));
        auto&& [faces, lgc] = build_infix(bdy);

        static size_type const expected_lgc[] = {logic::lopen,
                                                 0u,
                                                 logic::land,
                                                 logic::lnot,
                                                 1u,
                                                 logic::land,
                                                 2u,
                                                 logic::lclose};
        static S const expected_faces[] = {S{0u}, S{1u}, S{4u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
        EXPECT_EQ("all(+0, -1, +4)", build_infix_string(tree_, bdy));
    }

    // Imply inside boundary
    replace_and_simplify(&tree_, bdy, True{});

    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: ->{0}, 3: ->{1}, 4: ->{0}, 5: surface 2, 6: not{5}, 7: ->{6}, 8: surface 3, 9: not{8}, 10: ->{9}, 11: ->{5}, 12: all{5,9}, 13: ->{0}, 14: ->{0}, 15: ->{0}, })",
        to_string(tree_));

    // Test infix builder with remapping
    {
        auto remapped_surf = calc_surfaces(tree_);
        static S const expected_remapped_surf[] = {S{2u}, S{3u}};
        EXPECT_VEC_EQ(expected_remapped_surf, remapped_surf);
        auto&& [faces, lgc] = build_infix(shell, &remapped_surf);

        static size_type const expected_lgc[]
            = {logic::lopen, 0u, logic::land, logic::lnot, 1u, logic::lclose};
        static S const expected_faces[] = {S{0u}, S{1u}};
        EXPECT_VEC_EQ(expected_lgc, lgc);
        EXPECT_VEC_EQ(expected_faces, faces);
    }
}

//! Polycone didn't correctly get replaced with 'true' due to union
TEST_F(CsgTreeUtilsTest, tilecal_polycone_bug)
{
    EXPECT_EQ(N{2}, this->insert(Surface{S{0}}));  // lower z
    EXPECT_EQ(N{3}, this->insert(Surface{S{1}}));  // middle z
    EXPECT_EQ(N{4}, this->insert(Negated{N{3}}));  // below middle z
    EXPECT_EQ(N{5}, this->insert(Surface{S{2}}));  // cone
    EXPECT_EQ(N{6}, this->insert(Joined{op_and, {N{2}, N{4}, N{5}}}));  // lower
                                                                        // cone
    EXPECT_EQ(N{7}, this->insert(Surface{S{3}}));  // top z
    EXPECT_EQ(N{8}, this->insert(Negated{N{7}}));  // below top z
    EXPECT_EQ(N{9}, this->insert(Surface{S{4}}));  // cone
    EXPECT_EQ(N{10},
              this->insert(Joined{op_and, {N{3}, N{8}, N{9}}}));  // upper
                                                                  // cone
    EXPECT_EQ(N{11}, this->insert(Joined{op_or, {N{6}, N{10}}}));  // cone
    EXPECT_EQ(N{12}, this->insert(Negated{N{11}}));  // exterior
    EXPECT_EQ(N{13}, this->insert(Surface{S{5}}));  // muon box
    EXPECT_EQ(N{14}, this->insert(Negated{N{13}}));  // outside muon box
    EXPECT_EQ(N{15}, this->insert(Joined{op_and, {N{14}, N{11}}}));  // interior

    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: surface 2, 6: all{2,4,5}, 7: surface 3, 8: not{7}, 9: surface 4, 10: all{3,8,9}, 11: any{6,10}, 12: not{11}, 13: surface 5, 14: not{13}, 15: all{11,14}, })",
        to_string(tree_));

    // Imply inside boundary
    replace_and_simplify(&tree_, N{12}, True{});

    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: surface 2, 6: ->{1}, 7: surface 3, 8: not{7}, 9: surface 4, 10: ->{1}, 11: ->{1}, 12: ->{0}, 13: surface 5, 14: not{13}, 15: ->{1}, })",
        to_string(tree_));
}

//! Cylinder segment didn't correctly propagate logic
TEST_F(CsgTreeUtilsTest, tilecal_barrel_bug)
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
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: surface 2, 6: not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: all{2,4,9}, 11: not{10}, 12: surface 4, 13: surface 5, 14: all{12,13}, 15: all{2,4,6,11,12,13}, 16: not{15}, 17: surface 6, 18: surface 7, 19: not{18}, 20: all{6,17,19}, 21: all{9,17,19}, 22: not{21}, 23: surface 8, 24: surface 9, 25: all{23,24}, 26: all{6,17,19,22,23,24}, 27: not{26}, 28: all{2,4,6,11,12,13,27}, })",
        to_string(tree_));

    EXPECT_EQ("!all(+0, -1, -2, !all(+0, -1, -3), +4, +5)",
              build_infix_string(tree_, N{16}));
    replace_and_simplify(&tree_, N{16}, False{});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: ->{0}, 3: ->{1}, 4: ->{0}, 5: ->{1}, 6: ->{0}, 7: ->{0}, 8: ->{0}, 9: ->{1}, 10: ->{1}, 11: ->{0}, 12: ->{0}, 13: ->{0}, 14: ->{0}, 15: ->{0}, 16: ->{1}, 17: surface 6, 18: surface 7, 19: not{18}, 20: all{17,19}, 21: ->{1}, 22: ->{0}, 23: surface 8, 24: surface 9, 25: all{23,24}, 26: all{17,19,23,24}, 27: not{26}, 28: ->{27}, })",
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
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: not{3}, 6: any{4,5}, })",
        to_string(tree_));

    // Imply inside neither
    replace_and_simplify(&tree_, inside_a_or_b, False{});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: ->{0}, 3: ->{0}, 4: ->{1}, 5: ->{1}, 6: ->{1}, })",
        to_string(tree_));
}

TEST_F(CsgTreeUtilsTest, replace_union_2)
{
    auto a = this->insert(S{0});
    auto b = this->insert(S{1});
    this->insert(Negated{b});
    auto outside_a_or_b = this->insert(Joined{op_or, {a, b}});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: any{2,3}, })",
        to_string(tree_));

    // Imply !(a | b) -> a & b
    replace_and_simplify(&tree_, outside_a_or_b, False{});
    EXPECT_EQ("{0: true, 1: not{0}, 2: ->{1}, 3: ->{1}, 4: ->{0}, 5: ->{1}, }",
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

TEST_F(CsgTreeUtilsTest, transform_negated_joins)
{
    auto s0 = this->insert(Surface{S{0}});
    auto s1 = this->insert(Surface{S{1}});
    auto n0 = this->insert(Negated{s1});
    auto j0 = this->insert(Joined{op_and, {s0, n0}});

    // Check a well-formed tree
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, })",
        to_string(tree_));
    // Check that we have a noop
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{3},
            N{4},
            N{5},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    auto n1 = this->insert(Negated{j0});

    // Check a well-formed tree
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, })",
        to_string(tree_));
    // Check an easy case with just a single negated operand
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: any{3,4}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{},
            N{},
            N{5},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    auto j1 = this->insert(Joined{op_or, {s0, n0}});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, })",
        to_string(tree_));

    // Check that the non-negated operand maps to correct new node_ids and
    // that not{2} is not deleted
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: not{4}, 6: any{3,4}, 7: any{2,5}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{5},
            N{},
            N{6},
            N{7},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    auto n2 = this->insert(Negated{j1});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, 8: not{7}, })",
        to_string(tree_));
    // Check that the two operands are transformed, removing dangling
    // operators
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: any{3,4}, 6: all{3,4}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{},
            N{},
            N{5},
            N{},
            N{6},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    auto s2 = this->insert(Surface{S{2}});
    this->insert(Negated{s2});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, 8: not{7}, 9: surface 2, 10: not{9}, })",
        to_string(tree_));
    // Check that disjoint trees are correctly handled
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: any{3,4}, 6: all{3,4}, 7: surface 2, 8: not{7}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{},
            N{},
            N{5},
            N{},
            N{6},
            N{7},
            N{8},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    auto j2 = this->insert(Joined{op_and, {j0, j1}});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, 8: not{7}, 9: surface 2, 10: not{9}, 11: all{2,4,7}, })",
        to_string(tree_));

    // Add a non-transformed operand with suboperands
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: not{4}, 6: any{3,4}, 7: all{3,4}, 8: any{2,5}, 9: surface 2, 10: not{9}, 11: all{2,5,8}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{5},
            N{},
            N{6},
            N{8},
            N{7},
            N{9},
            N{10},
            N{11},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    auto n3 = this->insert(Negated{j2});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, 8: not{7}, 9: surface 2, 10: not{9}, 11: all{2,4,7}, 12: not{11}, })",
        to_string(tree_));

    // Top-level operand is negated and should be simplified, no need to
    // duplicate intermediary Joined nodes
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: any{3,4}, 6: all{3,4}, 7: surface 2, 8: not{7}, 9: any{3,4,6}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{},
            N{},
            N{5},
            N{},
            N{6},
            N{7},
            N{8},
            N{},
            N{9},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    auto j3 = this->insert(Joined{op_and, {n1, n2, n3}});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, 8: not{7}, 9: surface 2, 10: not{9}, 11: all{2,4,7}, 12: not{11}, 13: all{6,8,12}, })",
        to_string(tree_));

    // Top-level joined has Negated{Joined{}} chldrens
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: any{3,4}, 6: all{3,4}, 7: surface 2, 8: not{7}, 9: any{3,4,6}, 10: all{3,4,5,9}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{},
            N{},
            N{5},
            N{},
            N{6},
            N{7},
            N{8},
            N{},
            N{9},
            N{10},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    // Check a well-formed tree
    this->insert(Negated{j3});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: not{5}, 7: any{2,4}, 8: not{7}, 9: surface 2, 10: not{9}, 11: all{2,4,7}, 12: not{11}, 13: all{6,8,12}, 14: not{13}, })",
        to_string(tree_));

    // Complex case with a negated join with negated join as children
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: not{4}, 6: any{3,4}, 7: all{2,5}, 8: all{3,4}, 9: any{2,5}, 10: surface 2, 11: not{10}, 12: any{3,4,8}, 13: all{2,5,9}, 14: any{2,5,7,13}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{5},
            N{7},
            N{6},
            N{9},
            N{8},
            N{10},
            N{11},
            N{13},
            N{12},
            N{},
            N{14},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    tree_ = {};

    s0 = this->insert(S{0});
    s1 = this->insert(S{1});
    n0 = this->insert(Negated{s0});
    n1 = this->insert(Negated{s1});
    j0 = this->insert(Joined{op_or, {n0, n1}});
    n3 = this->insert(Negated{j0});
    s2 = this->insert(S{2});
    auto n4 = this->insert(Negated{s2});
    this->insert(Joined{op_and, {n3, n4}});
    // Check a well-formed tree
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: not{3}, 6: any{4,5}, 7: not{6}, 8: surface 2, 9: not{8}, 10: all{7,9}, })",
        to_string(tree_));

    // Complex case with a negated join with negated children
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: all{2,3}, 5: surface 2, 6: not{5}, 7: all{2,3,6}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{3},
            N{},
            N{},
            N{},
            N{4},
            N{5},
            N{6},
            N{7},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }

    tree_ = {};
    s0 = this->insert(S{0});
    s1 = this->insert(S{1});
    n0 = this->insert(Negated{s0});
    n1 = this->insert(Negated{s1});
    this->insert(Joined{op_and, {n0, n1}});
    j0 = this->insert(Joined{op_or, {n0, n1}});
    this->insert(Negated{j0});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: not{3}, 6: all{4,5}, 7: any{4,5}, 8: not{7}, })",
        to_string(tree_));
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: not{3}, 6: all{4,5}, 7: all{2,3}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{3},
            N{4},
            N{5},
            N{6},
            N{},
            N{7},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }
}

TEST_F(CsgTreeUtilsTest, transform_negated_joins_with_volumes)
{
    auto s0 = this->insert(S{0});
    auto s1 = this->insert(S{1});
    auto n0 = this->insert(Negated{s0});
    auto n1 = this->insert(Negated{s1});
    auto j0 = this->insert(Joined{op_or, {n0, n1}});
    auto n3 = this->insert(Negated{j0});
    auto s2 = this->insert(S{2});
    auto n4 = this->insert(Negated{s2});
    auto j1 = this->insert(Joined{op_and, {n3, n4}});
    tree_.insert_volume(j0);
    tree_.insert_volume(j1);
    tree_.insert_volume(n3);
    // Check a well-formed tree
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: not{3}, 6: any{4,5}, 7: not{6}, 8: surface 2, 9: not{8}, 10: all{7,9}, })",
        to_string(tree_));

    // Complex case with a negated join with negated children
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{2}, 5: not{3}, 6: all{2,3}, 7: any{4,5}, 8: surface 2, 9: not{8}, 10: all{2,3,9}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{3},
            N{4},
            N{5},
            N{7},
            N{6},
            N{8},
            N{9},
            N{10},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
        constexpr N expected_volumes[]{
            N{7},
            N{10},
            N{6},
        };
        EXPECT_VEC_EQ(expected_volumes, simplified.tree.volumes());
    }
    // Check that the new volumes map to the correct node

    tree_ = {};
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
    this->insert(Joined{op_and, {not_inner, outer_cyl}});
    auto bdy_outer = this->insert_surface(CCylZ{4.0});
    this->insert(Joined{op_and, {bdy_outer, mz, below_pz}});
    this->insert(Joined{op_and, {mz, below_pz}});
    tree_.insert_volume(inner_cyl);
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: not{4}, 6: surface 2, 7: not{6}, 8: any{3,4,6}, 9: all{2,5,7}, 10: surface 3, 11: not{10}, 12: all{2,5,11}, 13: all{2,5,8,11}, 14: surface 4, 15: all{2,5,14}, 16: all{2,5}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{5},
            N{6},
            N{7},
            N{9},
            N{10},
            N{11},
            N{12},
            N{8},
            N{13},
            N{14},
            N{15},
            N{16},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
        constexpr N expected_volumes[]{
            N{9},
        };
        EXPECT_VEC_EQ(expected_volumes, simplified.tree.volumes());
    }

    tree_ = {};
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

    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: surface 2, 6: not{5}, 7: all{2,4,6}, 8: surface 3, 9: not{8}, 10: all{2,4,9}, 11: not{10}, 12: surface 4, 13: surface 5, 14: all{12,13}, 15: all{2,4,6,11,12,13}, 16: not{15}, 17: surface 6, 18: surface 7, 19: not{18}, 20: all{6,17,19}, 21: all{9,17,19}, 22: not{21}, 23: surface 8, 24: surface 9, 25: all{23,24}, 26: all{6,17,19,22,23,24}, 27: not{26}, 28: all{2,4,6,11,12,13,27}, })",
        to_string(tree_));

    tree_.insert_volume(N{16});
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: not{4}, 6: surface 2, 7: not{6}, 8: all{2,5,7}, 9: surface 3, 10: not{9}, 11: any{3,4,9}, 12: all{2,5,10}, 13: surface 4, 14: not{13}, 15: surface 5, 16: not{15}, 17: all{13,15}, 18: any{3,4,6,12,14,16}, 19: surface 6, 20: not{19}, 21: surface 7, 22: not{21}, 23: all{7,19,22}, 24: any{9,20,21}, 25: all{10,19,22}, 26: surface 8, 27: not{26}, 28: surface 9, 29: not{28}, 30: all{26,28}, 31: any{6,20,21,25,27,29}, 32: all{2,5,7,11,13,15,31}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[] = {
            N{0},  N{1},  N{2},  N{4},  N{5},  N{6},  N{7},  N{8},
            N{9},  N{10}, N{12}, N{11}, N{13}, N{15}, N{17}, N{},
            N{18}, N{19}, N{21}, N{22}, N{23}, N{25}, N{24}, N{26},
            N{28}, N{30}, N{},   N{31}, N{32},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
        constexpr N expected_volumes[]{
            N{18},
        };
        EXPECT_VEC_EQ(expected_volumes, simplified.tree.volumes());
    }
}

TEST_F(CsgTreeUtilsTest, transform_negated_joins_with_aliases)
{
    auto s0 = this->insert(Surface{S{0}});
    auto s1 = this->insert(Surface{S{1}});
    auto n0 = this->insert(Negated{s1});
    auto j0 = this->insert(Joined{op_and, {s0, n0}});
    auto a0 = this->insert(Aliased{j0});
    this->insert(Negated{a0});
    EXPECT_EQ(
        R"({0: true, 1: not{0}, 2: surface 0, 3: surface 1, 4: not{3}, 5: all{2,4}, 6: ->{5}, 7: not{5}, })",
        to_string(tree_));
    {
        auto simplified = transform_negated_joins(tree_);
        EXPECT_EQ(
            R"({0: true, 1: not{0}, 2: surface 0, 3: not{2}, 4: surface 1, 5: not{4}, 6: any{3,4}, 7: all{2,5}, })",
            to_string(simplified.tree));
        constexpr N expected_new_nodes[]{
            N{0},
            N{1},
            N{2},
            N{4},
            N{5},
            N{},
            N{7},
            N{6},
        };
        EXPECT_VEC_EQ(expected_new_nodes, simplified.new_nodes);
    }
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
