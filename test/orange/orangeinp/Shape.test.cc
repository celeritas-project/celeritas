//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Shape.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/Shape.hh"

#include "orange/orangeinp/detail/CsgUnitBuilder.hh"
#include "orange/orangeinp/detail/VolumeBuilder.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

using namespace celeritas::orangeinp::detail::test;

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
class ShapeTest : public ::celeritas::test::Test
{
  private:
    using Unit = orangeinp::detail::CsgUnit;
    using UnitBuilder = orangeinp::detail::CsgUnitBuilder;
    using State = orangeinp::detail::ConvexSurfaceState;
    using Tol = UnitBuilder::Tol;

  protected:
    //! Access unit for testing results
    Unit const& unit() const { return unit_; }

    // Construct a volume from an object
    LocalVolumeId build_volume(ObjectInterface const& s);

  private:
    Unit unit_;
    UnitBuilder unit_builder_{&unit_, Tol::from_relative(1e-4)};
};

//---------------------------------------------------------------------------//
/*!
 * Construct a shape.
 */
LocalVolumeId ShapeTest::build_volume(ObjectInterface const& s)
{
    detail::VolumeBuilder vb{&unit_builder_};
    auto final_node = s.build(vb);
    return unit_builder_.insert_volume(final_node);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ShapeTest, single)
{
    auto box = this->build_volume(BoxShape{"box", Real3{1, 2, 3}});
    EXPECT_EQ(LocalVolumeId{0}, box) << box.unchecked_get();

    static char const* const expected_surface_strings[] = {"Plane: x=-1",
                                                           "Plane: x=1",
                                                           "Plane: y=-2",
                                                           "Plane: y=2",
                                                           "Plane: z=-3",
                                                           "Plane: z=3"};
    static char const* const expected_md_strings[] = {"",
                                                      "",
                                                      "box@mx",
                                                      "box@px",
                                                      "",
                                                      "box@my",
                                                      "box@py",
                                                      "",
                                                      "box@mz",
                                                      "box@pz",
                                                      "",
                                                      "box"};
    static int const expected_volume_nodes[] = {11};
    static char const* const expected_fill_strings[] = {"<UNASSIGNED>"};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["S",1],["~",3],["S",2],["S",3],["~",6],["S",4],["S",5],["~",9],["&",[2,4,5,7,8,10]]])json";

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
    EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
    EXPECT_EQ(NodeId{}, u.exterior);
}

TEST_F(ShapeTest, multiple)
{
    auto box = this->build_volume(BoxShape{"box", Real3{1, 1, 2}});
    auto cone = this->build_volume(Shape{"cone", Cone{{1.0, 0.5}, 2.0}});
    auto cyl = this->build_volume(Shape{"cyl", Cylinder{1.0, 2.0}});

    EXPECT_EQ(LocalVolumeId{0}, box) << box.unchecked_get();
    EXPECT_EQ(LocalVolumeId{1}, cone) << cone.unchecked_get();
    EXPECT_EQ(LocalVolumeId{2}, cyl) << cyl.unchecked_get();

    static char const* const expected_surface_strings[] = {
        "Plane: x=-1",
        "Plane: x=1",
        "Plane: y=-1",
        "Plane: y=1",
        "Plane: z=-2",
        "Plane: z=2",
        "Cone z: t=0.125 at {0,0,6}",
        "Cyl z: r=1",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "box@mx",
        "box@px",
        "",
        "box@my",
        "box@py",
        "",
        "box@mz,cone@mz,cyl@mz",
        "box@pz,cone@pz,cyl@pz",
        "",
        "box",
        "cone@kz",
        "",
        "cone",
        "cyl@cz",
        "",
        "cyl",
    };
    static int const expected_volume_nodes[] = {11, 14, 17};
    static char const* const expected_fill_strings[]
        = {"<UNASSIGNED>", "<UNASSIGNED>", "<UNASSIGNED>"};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["S",1],["~",3],["S",2],["S",3],["~",6],["S",4],["S",5],["~",9],["&",[2,4,5,7,8,10]],["S",6],["~",12],["&",[8,10,13]],["S",7],["~",15],["&",[8,10,16]]])json";

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
    EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
    EXPECT_EQ(NodeId{}, u.exterior);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
