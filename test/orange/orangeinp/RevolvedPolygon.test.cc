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
};

//---------------------------------------------------------------------------//
/*
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
 *
 */
TEST_F(RevolvedPolygonTest, one_subregion)
{
    VecReal2 polygon{{0, 0}, {3, 0}, {3, 2}, {0, 2}};

    this->build_volume(RevolvedPolygon{"rp", std::move(polygon)});

    static char const* const expected_surface_strings[]
        = {"Plane: z=0", "Plane: z=2", "Cyl z: r=3"};

    static char const* const expected_volume_strings[] = {"all(+0, -1, -2)"};

    static char const* const expected_md_strings[] = {"",
                                                      "",
                                                      "rp@0.0.0.mz",
                                                      "rp@0.0.0.pz",
                                                      "",
                                                      "rp@0.0.0.cz",
                                                      "",
                                                      "rp@0.0.0,rp@0.0.ou"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
/*
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

    this->build_volume(RevolvedPolygon{"rp", std::move(polygon)});

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

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
/*
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

    this->build_volume(RevolvedPolygon{"rp", std::move(polygon)});

    static char const* const expected_surface_strings[]
        = {"Plane: z=0",
           "Plane: z=2",
           "Cone z: t=0.5 at {0,0,0}",
           "Cyl z: r=3",
           "Plane: z=1.5",
           "Cone z: t=0.8 at {0,0,0}",
           "Cone z: t=0.4 at {0,0,4.5}"};
    static char const* const expected_volume_strings[]
        = {"all(+0, -1, -3, !all(+0, -1, -2), !all(!all(+0, -1, -2), "
           "any(all(+0, -4, -5), all(-1, +4, -6))))"};

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

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
/*
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

    this->build_volume(RevolvedPolygon{"rp", std::move(polygon)});

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
    static char const* const expected_volume_strings[]
        = {"all(+0, -1, -2, !all(+0, -1, -3), !all(-1, +4, -6, !all(-1, +4, "
           "-5), !all(+4, -7, -8, !all(+4, -7, -9))))"};

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

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
