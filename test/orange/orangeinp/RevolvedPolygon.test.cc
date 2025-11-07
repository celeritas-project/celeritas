//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/RevolvedPolygon.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/RevolvedPolygon.hh"

#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Solid.hh"
#include "orange/orangeinp/Transformed.hh"
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"
#include "orange/orangeinp/detail/SenseEvaluator.hh"

#include "CsgTestUtils.hh"
#include "ObjectTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{

//---------------------------------------------------------------------------//
class RevolvedPolygonTest : public ObjectTestBase
{
  protected:
    using VecReal2 = RevolvedPolygon::VecReal2;

    Tol tolerance() const override { return Tol::from_default(); }

    SignedSense eval_sense(Real3 const& pos)
    {
        CELER_ASSERT(vol_id_);
        auto const& u = this->unit();
        CELER_ASSERT(vol_id_ < u.tree.volumes().size());
        auto n = u.tree.volumes()[vol_id_.get()];
        detail::SenseEvaluator eval_sense(u.tree, u.surfaces, pos);
        return eval_sense(n);
    };

    /// DATA  ///
    LocalVolumeId vol_id_;
};

//---------------------------------------------------------------------------//
/* Test the simplest case: a single subregion.
 \verbatim
    3 _
       |
    2 _|______________
 z     |              |
    1 _|              |
       |              |
    0 _|______________|__________
       |    |    |    |    |    |
       0    1    2    3    4    5
                   r
 \endverbatim
 */
TEST_F(RevolvedPolygonTest, one_subregion)
{
    VecReal2 polygon{{0, 0}, {3, 0}, {3, 2}, {0, 2}};

    vol_id_ = this->build_volume(
        RevolvedPolygon{"rp", std::move(polygon), EnclosedAzi{}});

    static char const* const expected_surface_strings[]
        = {"Plane: z=0", "Plane: z=2", "Cyl z: r=3"};

    static char const* const expected_volume_strings[] = {"all(+0, -1, -2)"};

    static char const* const expected_md_strings[] = {
        "",
        "",
        "rp@0.0.0.mz",
        "rp@0.0.0.pz",
        "",
        "rp@0.0.0.cz",
        "",
        "rp@0.0.0,rp@0.0.ou",
    };

    static char const* const expected_bound_strings[] = {
        R"(7: {{{-2.12,-2.12,0}, {2.12,2.12,2}}, {{-3,-3,0}, {3,3,2}}})",
    };

    // Test construction
    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));

    // Test senses
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0, 0, 1}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({2, 2, 1}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({-2, -2, 1}));

    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 0, 0}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({3, 0, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 3, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, -3, 1}));

    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 0, -1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({3.1, 0, 1}));
}

//---------------------------------------------------------------------------//
/* Test a single subregion with a restricted angle
 \verbatim
    3 _
       |
    2 _|______________
 z     |              |
    1 _|              |   With cos(theta) >= 0
       |              |
    0 _|______________|__________
       |    |    |    |    |    |
       0    1    2    3    4    5
                   r
 \endverbatim
 */
TEST_F(RevolvedPolygonTest, one_subregion_with_enclosed)
{
    VecReal2 polygon{{0, 0}, {3, 0}, {3, 2}, {0, 2}};

    vol_id_ = this->build_volume(RevolvedPolygon{
        "rp", std::move(polygon), EnclosedAzi{Turn{-0.25}, Turn{0.25}}});

    static char const* const expected_surface_strings[]
        = {"Plane: z=0", "Plane: z=2", "Cyl z: r=3", "Plane: x=0"};

    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, +3)",
    };

    static char const* const expected_md_strings[] = {
        "",
        "",
        "rp@0.0.0.mz",
        "rp@0.0.0.pz",
        "",
        "rp@0.0.0.cz",
        "",
        "rp@0.0.0,rp@0.0.ou",
        "rp@awm,rp@awp,rp@azi",
        "rp@restricted",
    };

    static char const* const expected_bound_strings[] = {
        "7: {{{-2.12,-2.12,0}, {2.12,2.12,2}}, {{-3,-3,0}, {3,3,2}}}",
        "8: {{{0,-inf,-inf}, {inf,inf,inf}}, {{0,-inf,-inf}, {inf,inf,inf}}}",
        "9: {{{0,-2.12,0}, {2.12,2.12,2}}, {{0,-3,0}, {3,3,2}}}"};

    // Test construction
    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));

    // Test senses
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0.1, 0.1, 1}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({2, 2, 1}));

    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 0, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 0, 0}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({3, 0, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 3, 1}));

    EXPECT_EQ(SignedSense::outside, this->eval_sense({-0.1, -0.1, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({-2, -2, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 0, -1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({3.1, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, -3.1, 1}));
}

//---------------------------------------------------------------------------//
/* Test two-subregion case consisting of a cone subtracted from a cylinder.
 \verbatim
    3 _
       |
    2 _|    __________
 z     |   /          |
    1 _|  /           |
       | /            |
    0 _|/_____________|__________
       |    |    |    |    |    |
       0    1    2    3    4    5
                   r
 \endverbatim
 */
TEST_F(RevolvedPolygonTest, two_subregion)
{
    VecReal2 polygon{{1, 2}, {0, 0}, {3, 0}, {3, 2}};

    vol_id_ = this->build_volume(
        RevolvedPolygon{"rp", std::move(polygon), EnclosedAzi{}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=0", "Plane: z=2", "Cone z: t=0.5 at {0,0,0}", "Cyl z: r=3"};
    static char const* const expected_volume_strings[]
        = {"all(+0, -1, -3, !all(+0, -1, -2))"};

    static char const* const expected_md_strings[] = {
        "",
        "",
        "rp@0.0.0.mz,rp@0.0.1.mz",
        "rp@0.0.0.pz,rp@0.0.1.pz",
        "",
        "rp@0.0.0.kz",
        "",
        "rp@0.0.0,rp@0.0.iu",
        "rp@0.0.1.cz",
        "",
        "rp@0.0.1,rp@0.0.ou",
        "rp@0.0.nui",
        "rp@0.0.d",
    };

    static char const* const expected_bound_strings[] = {
        "7: {{{-0.354,-0.354,1}, {0.354,0.354,2}}, {{-1,-1,0}, {1,1,2}}}",
        "10: {{{-2.12,-2.12,0}, {2.12,2.12,2}}, {{-3,-3,0}, {3,3,2}}}",
        "~11: {{{-0.354,-0.354,1}, {0.354,0.354,2}}, {{-1,-1,0}, {1,1,2}}}",
        "12: {{{-1,-1,0}, {1,1,2}}, {{-3,-3,0}, {3,3,2}}}",
    };

    // Test construction
    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));

    // Test senses
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0.1, 0.1, 0.1}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({2, 2, 1}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({-2, -2, 1}));

    EXPECT_EQ(SignedSense::on, this->eval_sense({0.5, 0, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 0, 0}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({3, 0, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 3, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, -3, 1}));

    EXPECT_EQ(SignedSense::outside, this->eval_sense({0.45, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({-0.45, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 0, -1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({3.1, 0, 1}));
}

//---------------------------------------------------------------------------//
/* test case with a single concave region.
 \verbatim
    3 _
       |
    2 _|    __________
 z     |    \         |
    1 _|    /         |
       |  /           |
    0 _|/_____________|__________
       |    |    |    |    |    |
       0    1    2    3    4    5
                   r
 \endverbatim
 */
TEST_F(RevolvedPolygonTest, two_levels)
{
    VecReal2 polygon{{1, 2}, {1.2, 1.5}, {0, 0}, {3, 0}, {3, 2}};

    vol_id_ = this->build_volume(
        RevolvedPolygon{"rp", std::move(polygon), EnclosedAzi{}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=0",
        "Plane: z=2",
        "Cone z: t=0.5 at {0,0,0}",
        "Cyl z: r=3",
        "Plane: z=1.5",
        "Cone z: t=0.8 at {0,0,0}",
        "Cone z: t=0.4 at {0,0,4.5}",
    };
    static char const* const expected_volume_strings[] = {
        R"(all(+0, -1, -3, !all(+0, -1, -2), !all(!all(+0, -1, -2), any(all(+0, -4, -5), all(-1, +4, -6)))))",
    };

    static char const* const expected_md_strings[] = {
        "",
        "",
        "rp@0.0.0.mz,rp@0.0.1.mz,rp@1.0.0.mz,rp@1.0.2.mz",
        "rp@0.0.0.pz,rp@0.0.1.pz,rp@1.0.1.pz,rp@1.0.2.pz",
        "",
        "rp@0.0.0.kz,rp@1.0.2.kz",
        "",
        "rp@0.0.0,rp@0.0.iu,rp@1.0.2,rp@1.0.iu",
        "rp@0.0.1.cz",
        "",
        "rp@0.0.1,rp@0.0.ou",
        "rp@0.0.nui,rp@1.0.nui",
        "rp@0.0.d",
        "rp@1.0.0.pz,rp@1.0.1.mz",
        "",
        "rp@1.0.0.kz",
        "",
        "rp@1.0.0",
        "rp@1.0.1.kz",
        "",
        "rp@1.0.1",
        "rp@1.0.ou",
        "rp@0.cu,rp@1.0.d",
        "rp@0.ncu",
        "rp@0.d",
    };

    static char const* const expected_bound_strings[] = {
        "7: {{{-0.354,-0.354,1}, {0.354,0.354,2}}, {{-1,-1,0}, {1,1,2}}}",
        "10: {{{-2.12,-2.12,0}, {2.12,2.12,2}}, {{-3,-3,0}, {3,3,2}}}",
        "~11: {{{-0.354,-0.354,1}, {0.354,0.354,2}}, {{-1,-1,0}, {1,1,2}}}",
        "12: {{{-1,-1,0}, {1,1,2}}, {{-3,-3,0}, {3,3,2}}}",
        R"(17: {{{-0.424,-0.424,0.75}, {0.424,0.424,1.5}}, {{-1.2,-1.2,0}, {1.2,1.2,1.5}}})",
        R"(20: {{{-0.707,-0.707,1.5}, {0.707,0.707,2}}, {{-1.2,-1.2,1.5}, {1.2,1.2,2}}})",
        R"(21: {{{-0.707,-0.707,1.5}, {0.707,0.707,2}}, {{-1.2,-1.2,0}, {1.2,1.2,2}}})",
        "22: {null, {{-1.2,-1.2,0}, {1.2,1.2,2}}}",
        "~23: {null, {{-1.2,-1.2,0}, {1.2,1.2,2}}}",
        "24: {null, {{-3,-3,0}, {3,3,2}}}",
    };

    // Test construction
    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));

    // Test senses
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0.1, 0.1, 0.01}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({2, 2, 1}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({-2, -2, 1}));

    EXPECT_EQ(SignedSense::inside, this->eval_sense({0.61, 0, 0.75}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 0, 0}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({3, 0, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 3, 1}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, -3, 1}));

    EXPECT_EQ(SignedSense::outside, this->eval_sense({0.59, 0, 0.75}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0.45, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({-0.45, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 0, -1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 0, 1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({3.1, 0, 1}));
}

//---------------------------------------------------------------------------//
/* These cases with nested concavity.
 \verbatim
   3 __  __ . . . . . . .  ____
      | |  |              |    |
   2 _| |  |     ____     |    |
 z    | |  |    |    |    |    |
   1 _| |  |____|. . |____|    |
      | |______________________|
   0 _|________________________
      |    |    |    |    |    |
      0    1    2    3    4    5
                  r
  \endverbatim
 */
TEST_F(RevolvedPolygonTest, three_levels)
{
    VecReal2 polygon{{5, 0.5},
                     {5, 3},
                     {4, 3},
                     {4, 1},
                     {3, 1},
                     {3, 2},
                     {2, 2},
                     {2, 1},
                     {1, 1},
                     {1, 3},
                     {0.33, 3},
                     {0.33, 0.5}};

    vol_id_ = this->build_volume(
        RevolvedPolygon{"rp", std::move(polygon), EnclosedAzi{}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=0.5",
        "Plane: z=3",
        "Cyl z: r=5",
        "Cyl z: r=0.33",
        "Plane: z=1",
        "Cyl z: r=1",
        "Cyl z: r=4",
        "Plane: z=2",
        "Cyl z: r=3",
        "Cyl z: r=2",
    };
    static char const* const expected_volume_strings[] = {
        R"(all(+0, -1, -2, !all(+0, -1, -3), !all(-1, +4, -6, !all(-1, +4, -5), !all(+4, -7, -8, !all(+4, -7, -9)))))",
    };

    static char const* const expected_md_strings[] = {
        "",
        "",
        "rp@0.0.0.mz,rp@0.0.1.mz",
        "rp@0.0.0.pz,rp@0.0.1.pz,rp@1.0.0.pz,rp@1.0.1.pz",
        "",
        "rp@0.0.0.cz",
        "",
        "rp@0.0.0,rp@0.0.ou",
        "rp@0.0.1.cz",
        "",
        "rp@0.0.1,rp@0.0.iu",
        "rp@0.0.nui",
        "rp@0.0.d",
        "rp@1.0.0.mz,rp@1.0.1.mz,rp@2.0.0.mz,rp@2.0.1.mz",
        "rp@1.0.0.cz",
        "",
        "rp@1.0.0,rp@1.0.iu",
        "rp@1.0.1.cz",
        "",
        "rp@1.0.1,rp@1.0.ou",
        "rp@1.0.nui",
        "rp@1.0.d",
        "rp@2.0.0.pz,rp@2.0.1.pz",
        "",
        "rp@2.0.0.cz",
        "",
        "rp@2.0.0,rp@2.0.ou",
        "rp@2.0.1.cz",
        "",
        "rp@2.0.1,rp@2.0.iu",
        "rp@2.0.nui",
        "rp@1.cu,rp@2.0.d",
        "rp@1.ncu",
        "rp@0.cu,rp@1.d",
        "rp@0.ncu",
        "rp@0.d",
    };

    static char const* const expected_bound_strings[] = {
        "7: {{{-3.54,-3.54,0.5}, {3.54,3.54,3}}, {{-5,-5,0.5}, {5,5,3}}}",
        R"(10: {{{-0.233,-0.233,0.5}, {0.233,0.233,3}}, {{-0.33,-0.33,0.5}, {0.33,0.33,3}}})",
        R"(~11: {{{-0.233,-0.233,0.5}, {0.233,0.233,3}}, {{-0.33,-0.33,0.5}, {0.33,0.33,3}}})",
        "12: {{{-0.33,-0.33,0.5}, {0.33,0.33,3}}, {{-5,-5,0.5}, {5,5,3}}}",
        "16: {{{-0.707,-0.707,1}, {0.707,0.707,3}}, {{-1,-1,1}, {1,1,3}}}",
        "19: {{{-2.83,-2.83,1}, {2.83,2.83,3}}, {{-4,-4,1}, {4,4,3}}}",
        "~20: {{{-0.707,-0.707,1}, {0.707,0.707,3}}, {{-1,-1,1}, {1,1,3}}}",
        "21: {{{-1,-1,1}, {1,1,3}}, {{-4,-4,1}, {4,4,3}}}",
        "26: {{{-2.12,-2.12,1}, {2.12,2.12,2}}, {{-3,-3,1}, {3,3,2}}}",
        "29: {{{-1.41,-1.41,1}, {1.41,1.41,2}}, {{-2,-2,1}, {2,2,2}}}",
        "~30: {{{-1.41,-1.41,1}, {1.41,1.41,2}}, {{-2,-2,1}, {2,2,2}}}",
        "31: {{{-2,-2,1}, {2,2,2}}, {{-3,-3,1}, {3,3,2}}}",
        "~32: {{{-2,-2,1}, {2,2,2}}, {{-3,-3,1}, {3,3,2}}}",
        "33: {null, {{-4,-4,1}, {4,4,3}}}",
        "~34: {null, {{-4,-4,1}, {4,4,3}}}",
        "35: {null, {{-5,-5,0.5}, {5,5,3}}}",
    };

    // Test construction
    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));

    // Test senses
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0.6, 0, 2}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0, 2.5, 1.5}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({0, -2.5, 1.5}));
    EXPECT_EQ(SignedSense::inside, this->eval_sense({3.2, -3.2, 2.9}));

    EXPECT_EQ(SignedSense::on, this->eval_sense({4, 0, 0.5}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({-2, -2, 0.5}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0.7, 0, 3}));
    EXPECT_EQ(SignedSense::on, this->eval_sense({0, 2.5, 2}));

    EXPECT_EQ(SignedSense::outside, this->eval_sense({3, -3, 0.3}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0.3, 0, 2}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({1.5, 0, 2.1}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({-3.5, 0, 1.5}));
    EXPECT_EQ(SignedSense::outside, this->eval_sense({0, 2.5, 2.5}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
