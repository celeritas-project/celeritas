//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/SolidConverter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/SolidConverter.hh"

#include <initializer_list>
#include <random>
#include <G4BooleanSolid.hh>
#include <G4Box.hh>
#include <G4Cons.hh>
#include <G4CutTubs.hh>
#include <G4DisplacedSolid.hh>
#include <G4Ellipsoid.hh>
#include <G4EllipticalCone.hh>
#include <G4EllipticalTube.hh>
#include <G4ExtrudedSolid.hh>
#include <G4GenericPolycone.hh>
#include <G4GenericTrap.hh>
#include <G4Hype.hh>
#include <G4IntersectionSolid.hh>
#include <G4MultiUnion.hh>
#include <G4Orb.hh>
#include <G4Para.hh>
#include <G4Paraboloid.hh>
#include <G4PhysicalConstants.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4ReflectedSolid.hh>
#include <G4RotationMatrix.hh>
#include <G4ScaledSolid.hh>
#include <G4Sphere.hh>
#include <G4SubtractionSolid.hh>
#include <G4SystemOfUnits.hh>
#include <G4TessellatedSolid.hh>
#include <G4Tet.hh>
#include <G4ThreeVector.hh>
#include <G4Torus.hh>
#include <G4Trap.hh>
#include <G4Trd.hh>
#include <G4Tubs.hh>
#include <G4UnionSolid.hh>

#include "corecel/Constants.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/Turn.hh"
#include "geocel/random/UniformBoxDistribution.hh"
#include "orange/BoundingBoxUtils.hh"
#include "orange/g4org/Scaler.hh"
#include "orange/g4org/Transformer.hh"
#include "orange/orangeinp/CsgTestUtils.hh"
#include "orange/orangeinp/ObjectInterface.hh"
#include "orange/orangeinp/ObjectTestBase.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/orangeinp/detail/SenseEvaluator.hh"

#include "celeritas_test.hh"

using celeritas::orangeinp::detail::SenseEvaluator;
using CLHEP::cm;
using CLHEP::deg;
using CLHEP::halfpi;
using CLHEP::mm;
constexpr double half{0.5};

namespace celeritas
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, SignedSense s)
{
    return (os << to_cstring(s));
}

namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
SignedSense to_signed_sense(EInside inside)
{
    switch (inside)
    {
        case kOutside:
            return SignedSense::outside;
        case kSurface:
            return SignedSense::on;
        case kInside:
            return SignedSense::inside;
    }
    CELER_ASSERT_UNREACHABLE();
}

bool sense_equal(SignedSense lhs, SignedSense rhs)
{
    // Disagreeing about what's "on" is usually fine
    if (lhs == SignedSense::on || rhs == SignedSense::on)
        return true;
    return lhs == rhs;
}

G4ThreeVector to_geant(Real3 const& rv)
{
    return {rv[0], rv[1], rv[2]};
}

//---------------------------------------------------------------------------//
class SolidConverterTest : public ::celeritas::orangeinp::test::ObjectTestBase
{
  protected:
    Tol tolerance() const { return Tol::from_default(); }

    // Test the solid, the generated hierarchy, and points in space [cm]
    void build_and_test(G4VSolid const& solid,
                        std::string_view json_str = "null",
                        std::initializer_list<Real3> points = {});

    Scaler scale_{0.1};
    Transformer transform_{scale_};
    std::size_t num_samples_{4096};  //! Number of points to sample
};

//---------------------------------------------------------------------------//
void SolidConverterTest::build_and_test(G4VSolid const& solid,
                                        std::string_view json_str,
                                        std::initializer_list<Real3> points)
{
    SCOPED_TRACE(solid.GetName());

    // Recreate the converter at each step since the solid may be a
    // temporary rather than in a "store"
    SolidConverter convert{scale_, transform_};

    // Convert the object
    auto obj = convert(solid);
    CELER_ASSERT(obj);
    EXPECT_JSON_EQ(json_str, to_string(*obj));

    // Construct a volume from it
    auto vol_id = this->build_volume(*obj);
    auto const& u = this->unit();
    auto node = u.tree.volumes()[vol_id.get()];

    // Set up functions to calculate in/on/out
    CELER_ASSERT(vol_id < u.tree.volumes().size());
    auto calc_org_sense = [&u, node](Real3 const& pos) {
        SenseEvaluator eval_sense(u.tree, u.surfaces, pos);
        return eval_sense(node);
    };
    auto calc_g4_sense = [&solid,
                          inv_scale = 1 / scale_.value()](Real3 const& pos) {
        return to_signed_sense(solid.Inside(G4ThreeVector(
            inv_scale * pos[0], inv_scale * pos[1], inv_scale * pos[2])));
    };

    // Test user-supplied points [cm]
    for (Real3 const& r : points)
    {
        auto g4_sense = calc_g4_sense(r);
        auto org_sense = calc_org_sense(r);
        EXPECT_TRUE(sense_equal(g4_sense, org_sense))
            << "G4 " << g4_sense << " != ORANGE " << org_sense << " at " << r
            << " [cm]";
    }

    // Test random points
    auto const& bbox = [&u, node] {
        auto iter = u.regions.find(node);
        CELER_ASSERT(iter != u.regions.end());
        auto const& bounds = iter->second.bounds;
        CELER_ASSERT(!bounds.negated);
        return bounds.exterior;
    }();
    if (is_finite(bbox))
    {
        // Expand the bounding box and check points
        BoundingBoxBumper<real_type> bump_bb(
            {/* rel = */ 0.25, /* abs = */ 0.01});
        auto expanded_bbox = bump_bb(bbox);
        CELER_LOG(debug) << "Sampling '" << solid.GetName() << "' inside box "
                         << expanded_bbox;

        std::mt19937_64 rng;
        UniformBoxDistribution sample_box(expanded_bbox.lower(),
                                          expanded_bbox.upper());
        for ([[maybe_unused]] auto i : range(this->num_samples_))
        {
            auto r = sample_box(rng);
            EXPECT_EQ(calc_g4_sense(r), calc_org_sense(r))
                << "at " << r << " [cm]";
        }
    }
    else
    {
        CELER_LOG(warning) << "Not sampling '" << solid.GetName()
                           << "' due to non-finite bounding box " << bbox;
    }
}

//---------------------------------------------------------------------------//
// SOLID TESTS
// NOTE: keep these alphabetically ordered
//---------------------------------------------------------------------------//

TEST_F(SolidConverterTest, box)
{
    this->build_and_test(
        G4Box("Test Box", 20, 30, 40),
        R"json({"_type":"shape","interior":{"_type":"box","halfwidths":[2.0,3.0,4.0]},"label":"Test Box"})json",
        {{1, 0, 0}, {0, 3, 0}, {0, 0, 4.1}});
}

TEST_F(SolidConverterTest, cons)
{
    this->build_and_test(
        G4Cons("Solid TubeLike #1", 0, 50, 0, 50, 50, 0, 360),
        R"json({"_type":"shape","interior":{"_type":"cone","halfheight":5.0,"radii":[5.0,5.0]},"label":"Solid TubeLike #1"})json",
        {{0, 0, 4}, {0, 0, 5}, {0, 0, 6}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}});

    this->build_and_test(
        G4Cons("test10",
               20.0,
               80.0,
               60.0,
               140.0,
               1.0,
               0.17453292519943,
               5.235987755983),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.8611111111111125,"start":0.027777777777777308},"excluded":{"_type":"cone","halfheight":0.1,"radii":[2.0,6.0]},"interior":{"_type":"cone","halfheight":0.1,"radii":[8.0,14.0]},"label":"test10"})json");

    this->build_and_test(
        G4Cons(
            "aCone", 2 * cm, 6 * cm, 8 * cm, 14 * cm, 10 * cm, 10 * deg, 300 * deg),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.8611111111111112,"start":0.027777777777777776},"excluded":{"_type":"cone","halfheight":10.0,"radii":[2.0,8.0]},"interior":{"_type":"cone","halfheight":10.0,"radii":[6.0,14.0]},"label":"aCone"})json");
}

TEST_F(SolidConverterTest, displaced)
{
    // Daughter to parent: +x becomes +y
    Real3 const rot_axis = make_unit_vector(Real3{3, 4, 5});
    Turn const rot_turn{0.125};

    // Construct Geant4 matrix and transforms
    G4Transform3D transform(
        G4RotationMatrix(to_geant(rot_axis), native_value_from(rot_turn)),
        G4ThreeVector(10.0, 20.0, 30.0));

    G4Box box("box", 20, 30, 40);

    this->build_and_test(
        G4DisplacedSolid("boxd", &box, transform),
        R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[2.0,3.0,4.0]},"label":"box"},"transform":{"_type":"transformation","data":[0.7598275605729691,-0.42970562748477137,0.4878679656440358,0.5702943725152286,0.8008326112068523,-0.18284271247461906,-0.31213203435596426,0.41715728752538106,0.8535533905932738,1.0,2.0,3.0]}})json",
        {{1, 2, 3}, {2, 2, 3}, {3, 0, 0}});
}

TEST_F(SolidConverterTest, ellipsoid)
{
    this->build_and_test(
        G4Ellipsoid("with_trunc", 10 * cm, 20 * cm, 30 * cm, -1 * cm, 29 * cm),
        R"json({"_type":"truncated","planes":[{"axis":"z","position":29.0,"sense":"inside"},{"axis":"z","position":-1.0,"sense":"outside"}],"region":{"_type":"ellipsoid","radii":[10.0,20.0,30.0]}})json",
        {{0, 0, 0},
         {0, 0, -1.1},
         {0, 0, 29.1},
         {9.95, 19.95, 29.95},
         {10.05, 20.05, 30.05}});
    this->build_and_test(
        G4Ellipsoid("without_trunc", 10 * cm, 20 * cm, 30 * cm),
        R"json({"_type":"shape","interior":{"_type":"ellipsoid","radii":[10.0,20.0,30.0]},"label":"without_trunc"})json",
        {{0, 0, 0},
         {0, 1.9, 0},
         {0, 2.1, 0},
         {0, 0, 29.1},
         {9.95, 19.95, 29.95},
         {10.05, 20.05, 30.05}});
}

TEST_F(SolidConverterTest, ellipticalcylinder)
{
    this->build_and_test(
        G4EllipticalTube("testEllipticalCylinder", 10 * cm, 20 * cm, 30 * cm),
        R"json({"_type":"shape","interior":{"_type":"ellipticalcylinder","halfheight":30.0,"radii":[10.0,20.0]},"label":"testEllipticalCylinder"})json",
        {{0, 0, 0}, {0, 21, 0}, {0., 0, 31}, {1., 0, 0}});
}

TEST_F(SolidConverterTest, ellipticalcone)
{
    this->build_and_test(
        G4EllipticalCone("testEllipticalCone", 0.4, 0.8, 50, 25),
        R"json({"_type":"shape","interior":{"_type":"ellipticalcone","halfheight":2.5,"lower_radii":[3.0,6.0],"upper_radii":[1.0,2.0]},"label":"testEllipticalCone"})json",
        {{0, 0, 0}, {0, 0, 24.9}, {0., 0, -24.9}});
}

//---------------------------------------------------------------------------//
/*
 * Test xtru with 4 levels of concavity. Points are supplied in clockwise
 order,
 * as preferred by Geant4.
 \verbatim
                   7
 1                 |\
 \\                | \
  \ \      3  5    |  \
   \  \    /\/\    |   \
    \   \/   4  \  |    \ 8
     \   2        \|    /
      \            6   /
       \ 11           /
        \/\__________/
        0  10        9
 \endverbatim
 */
TEST_F(SolidConverterTest, extrudedsolid_concave)
{
    using ZSection = G4ExtrudedSolid::ZSection;

    // Setup G4Extruded solid construction commands
    std::vector<G4TwoVector> polygon = {{0, 0},
                                        {-0.3, 1},
                                        {0.15, 0.5},
                                        {0.4, 0.7},
                                        {0.45, 0.6},
                                        {0.5, 0.7},
                                        {0.8, 0.4},
                                        {0.9, 1.2},
                                        {1.2, 0.5},
                                        {1, 0},
                                        {0.1, 0},
                                        {0.05, 0.01}};

    ZSection bot(0, {0, 0}, 1);
    ZSection mid(1, {10, 5}, 0.5);
    ZSection top(2, {1, 2}, 1.5);
    std::vector<ZSection> z_sections{bot, mid, top};

    // Build and test, with 5 points near tricky corners explicitly tested
    this->build_and_test(
        G4ExtrudedSolid("testExtrudedSolid", polygon, z_sections),
        R"json({"_type":"stackedextrudedpolygon","polygon":[[0.005000000000000001,0.001],[0.010000000000000002,0.0],[0.1,0.0],[0.12,0.05],[0.09000000000000001,0.12],[0.08000000000000002,0.04000000000000001],[0.05,0.06999999999999999],[0.045000000000000005,0.06],[0.04000000000000001,0.06999999999999999],[0.015,0.05],[-0.03,0.1],[0.0,0.0]],"polyline":[[0.0,0.0,0.0],[1.0,0.5,0.1],[0.1,0.2,0.2]],"scaling":[1.0,0.5,1.5]})json",
        {{0.01, 0.011, 0.3},
         {0.39, 0.79, 1.5},
         {0.79, 0.39, 1.1},
         {0.81, 0.4, 0.3},
         {0.89, 1.18, 0.5}});
}

//---------------------------------------------------------------------------//
/*
 * Test that xtru yields an ExtrudedPolygon, not a StackedExtrudedPolygon when
 * a convex polygon is used with a one-segment polyline.
 */
TEST_F(SolidConverterTest, extrudedsolid_simple)
{
    using ZSection = G4ExtrudedSolid::ZSection;

    std::vector<G4TwoVector> polygon = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    ZSection bot(0, {0, 0}, 1);
    ZSection top(1, {1, 2}, 1.5);
    std::vector<ZSection> z_sections{bot, top};

    // Test 4 points near tricky corners
    this->build_and_test(
        G4ExtrudedSolid("testExtrudedSolid", polygon, z_sections),
        R"json({"_type":"shape","interior":{"_type":"extrudedpolygon","bot_line_segment_point":[0.0,0.0,0.0],"bot_scaling_factor":1.0,"polygon":[[0.1,0.0],[0.1,0.1],[0.0,0.1],[0.0,0.0]],"top_line_segment_point":[0.1,0.2,0.1],"top_scaling_factor":1.5},"label":"testExtrudedSolid"})json",
        {{0.5, 0.5, 0.5}, {-1, 0.5, 0.5}});
}

//---------------------------------------------------------------------------//
/*
 * Test GenericPolygon with 4 levels of concavity. Points are supplied in
 * clockwise order, as preferred by Geant4.
 \verbatim
                   7
 1                 |\
 \\                | \
  \ \      3  5    |  \
   \  \    /\/\    |   \
    \   \/   4  \  |    \ 8
     \   2        \|    /
      \            6   /
       \ 11           /
        \/\__________/
        0  10        9
 \endverbatim
 */
TEST_F(SolidConverterTest, generic_polycone)
{
    G4double phi_start = 0 * deg;
    G4double phi_end = 90 * deg;
    std::vector<G4double> r{
        0.3, 0.0, 0.45, 0.7, 0.75, 0.8, 1.1, 1.2, 1.5, 1.3, 0.4, 0.35};
    std::vector<G4double> z{
        -0.5, 0.5, 0.0, 0.2, 0.1, 0.2, -0.1, 0.7, 0.0, -0.5, -0.5, -0.49};

    // Test 5 points near tricky corners and 2 outside of the azimuthal range
    this->build_and_test(
        G4GenericPolycone("testGenericPolycone",
                          phi_start,
                          phi_end,
                          r.size(),
                          r.data(),
                          z.data()),
        R"json({"_type":"revolvedpolygon","enclosed_azi":{"start":0.0,"stop":0.25},"label":"testGenericPolycone","polygon":[[0.034999999999999996,-0.049],[0.04000000000000001,-0.05],[0.13,-0.05],[0.15000000000000002,0.0],[0.12,0.06999999999999999],[0.11000000000000001,-0.010000000000000002],[0.08000000000000002,0.020000000000000004],[0.07500000000000001,0.010000000000000002],[0.06999999999999999,0.020000000000000004],[0.045000000000000005,0.0],[0.0,0.05],[0.03,-0.05]]})json",
        {
            {0.01, 0.011, -0.2},
            {0.39, 0.79, 1.0},
            {0.79, 0.39, 0.6},
            {0.81, 0.4, -0.2},
            {0.89, 1.18, 0.0},
            {-0.81, 0.4, -0.2},
            {-0.81, -0.4, -0.2},
        });
}

TEST_F(SolidConverterTest, generictrap)
{
    this->build_and_test(G4GenericTrap("boxGenTrap",
                                       30,
                                       {{-10, -20},
                                        {-10, 20},
                                        {10, 20},
                                        {10, -20},
                                        {-10, -20},
                                        {-10, 20},
                                        {10, 20},
                                        {10, -20}}),
                         R"json({"_type":"shape","interior":{"_type":"genprism",
                "halfheight":3.0,
                "lower":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]],
                "upper":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]]},
                "label":"boxGenTrap"})json",
                         {{-1, -2, -3}, {1, 2, 3}, {1, 2, 4}});

    this->build_and_test(
        G4GenericTrap("trdGenTrap",
                      3,
                      {{-10, -20},
                       {-10, 20},
                       {10, 20},
                       {10, -20},
                       {-5, -10},
                       {-5, 10},
                       {5, 10},
                       {5, -10}}),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":0.3,
            "lower":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]],
            "upper":[[0.5,-1.0],[0.5,1.0],[-0.5,1.0],[-0.5,-1.0]]},
            "label":"trdGenTrap"})json",
        {{-1, -2, -4}, {-1, -2, -3}, {0.5, 1, 3}, {1, 1, 3}});

    this->build_and_test(
        G4GenericTrap("trap_GenTrap",
                      40,
                      {{-9, -20},
                       {-9, 20},
                       {11, 20},
                       {11, -20},
                       {-29, -40},
                       {-29, 40},
                       {31, 40},
                       {31, -40}}),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":4.0,
            "lower":[[1.1,-2.0],[1.1,2.0],[-0.9,2.0],[-0.9,-2.0]],
            "upper":[[3.1,-4.0],[3.1,4.0],[-2.9,4.0],[-2.9,-4.0]]},
            "label":"trap_GenTrap"})json",
        {{-1, -2, -4 - 1.e-6}, {-1, -2, -3}, {0.5, 1, 3}, {1, 1, 3}});

    // Most general genprism with twisted side faces
    this->build_and_test(
        G4GenericTrap("LArEMECInnerWheelAbsorber02",
                      10.625,
                      {{1.55857990922689, 302.468976599716},
                       {-1.73031296208306, 302.468976599716},
                       {-2.53451906396442, 609.918546236458},
                       {2.18738922312177, 609.918546236458},
                       {-11.9586196560814, 304.204253530802},
                       {-15.2556006134987, 304.204253530802},
                       {-31.2774318502685, 613.426120316623},
                       {-26.5391748405779, 613.426120316623}}),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":1.0625,"lower":[[0.218738922312177,60.99185462364581],[-0.253451906396442,60.99185462364581],[-0.173031296208306,30.246897659971598],[0.155857990922689,30.246897659971598]],"upper":[[-2.65391748405779,61.342612031662306],[-3.12774318502685,61.342612031662306],[-1.52556006134987,30.420425353080205],[-1.19586196560814,30.420425353080205]]},"label":"LArEMECInnerWheelAbsorber02"})json",
        {
            {51.2, 0.40, 7.76},
            {51.4, 0.51, 7.78},
        });

    // GenTrapTest, trap_uneven_twist
    this->build_and_test(
        G4GenericTrap("trap_uneven_twist",
                      10,
                      {
                          {-20, -10},
                          {-20, 10},
                          {20, 10},
                          {20, -10},
                          {-15, -5},
                          {-5, 5},
                          {15, 5},
                          {5, -5},
                      }),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":1.0,
            "lower":[[2.0,-1.0],[2.0,1.0],[-2.0,1.0],[-2.0,-1.0]],
            "upper":[[0.5,-0.5],[1.5,0.5],[-0.5,0.5],[-1.5,-0.5]]},
            "label":"trap_uneven_twist"})json",
        {
            {1.99, -0.99, -0.99},
            {0.49, -0.49, 0.99},
        });

    // GenTrapTest, trap_even_twist
    this->build_and_test(
        G4GenericTrap("trap_even_twist",
                      1,
                      {{-2, -1},
                       {-2, 1},
                       {2, 1},
                       {2, -1},
                       {-3, -1},
                       {-1, 1},
                       {3, 1},
                       {1, -1}}),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":0.1,"lower":[[0.2,-0.1],[0.2,0.1],[-0.2,0.1],[-0.2,-0.1]],"upper":[[0.1,-0.1],[0.3,0.1],[-0.1,0.1],[-0.3,-0.1]]},"label":"trap_even_twist"})json",
        {
            {1.99, -0.99, -0.99},
            {0.49, -0.49, 0.99},
        });
}

TEST_F(SolidConverterTest, intersectionsolid)
{
    G4Box b1("Test Box #1", 20, 30, 40);
    G4Box b2("Test Box #2", 10, 10, 10);
    G4RotationMatrix xRot;
    xRot.rotateZ(-pi * 0.5);
    G4Transform3D transform(xRot, G4ThreeVector(0, 10, 0));
    this->build_and_test(
        G4IntersectionSolid("b1Intersectionb2", &b1, &b2, transform),
        R"json({"_type":"all","daughters":[{"_type":"shape","interior":{"_type":"box","halfwidths":[2.0,3.0,4.0]},"label":"Test Box #1"},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[1.0,1.0,1.0]},"label":"Test Box #2"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,1.0,0.0,-1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,0.0,1.0,0.0]}}],"label":"b1Intersectionb2"})json",
        {{0, 0, 0}, {0, 0, 10}, {0, 1, 0}});
}

//---------------------------------------------------------------------------//
/*
 * Test G4MultiUnion with three transformed volumes.
  \verbatim
     y
   3_|     ____     _     ____
     |    |    | /     \ |    |
   2_|    |    |   v1    |    |
     |    | v2 | \     / | v3 |
   1_|    |    |    _    |    |
     |    |    |         |    |
   0_|____|____|_________|____|_________ x
     |    |    |    |    |    |    |    |
     |   -2   -1    0    1    2    3    4
  -1_|    |    |         |    |
     |    |    |         |    |
  -2_|    |    |         |    |
     |    |    |         |    |
  -3_|    |____|         |____|
  \endverbatim
 */
TEST_F(SolidConverterTest, multiunion)
{
    G4MultiUnion mu("multiunion");

    // Add v1
    G4Tubs v1("v1", 0, 1 * cm, 1 * cm, 0, 360 * deg);
    G4Transform3D t1(G4RotationMatrix{}, G4ThreeVector(0, 2 * cm, 0));
    mu.AddNode(v1, t1);

    // Define rotation matrix for v2 and v3, which we will define horizontally
    G4RotationMatrix r90;
    r90.rotateZ(90 * deg);

    // Add v2
    G4Box v2("v2", 3 * cm, 0.5 * cm, 1 * cm);
    G4Transform3D t2(r90, G4ThreeVector(1.5 * cm, 0, 0));
    mu.AddNode(v2, t2);

    // Add v3
    G4Box v3("v3", 3 * cm, 0.5 * cm, 1 * cm);
    G4Transform3D t3(r90, G4ThreeVector(-1.5 * cm, 0, 0));
    mu.AddNode(v3, t3);

    // Voxelize to complete
    mu.Voxelize();

    this->build_and_test(
        mu,
        R"json({"_type":"any","daughters":[{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"cylinder","halfheight":1.0,"radius":1.0},"label":"v1"},"transform":{"_type":"transformation","data":[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,2.0,0.0]}},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[3.0,0.5,1.0]},"label":"v2"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,-1.0,0.0,1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,1.5,0.0,0.0]}},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[3.0,0.5,1.0]},"label":"v3"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,-1.0,0.0,1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,-1.5,0.0,0.0]}}],"label":"multiunion"})json",
        {
            {0, 2, 0},
            {0, 2, 0.6},
            {0, 2, -0.6},
            {1.5, -2.9, 0},
            {1.5, -3.1, 0},
            {-1.9, -2.9, 0},
            {-2.1, -2.9, 0},
        });
}

TEST_F(SolidConverterTest, orb)
{
    this->build_and_test(
        G4Orb("Solid G4Orb", 50),
        R"json({"_type":"shape","interior":{"_type":"sphere","radius":5.0},"label":"Solid G4Orb"})json",
        {{0, 0, 0}, {0, 5.0, 0}, {10.0, 0, 0}});
}

TEST_F(SolidConverterTest, paraboloid)
{
    this->build_and_test(
        G4Paraboloid("testParaboloid", 5, 1, 2),
        R"json({"_type":"shape","interior":{"_type":"paraboloid","halfheight":0.5,"lower_radius":0.1,"upper_radius":0.2},"label":"testParaboloid"})json",
        {{0, 0, 0},
         {0, 1.99, 5},
         {0., 2.01, 5},
         {0.99, -0.99, -4.9},
         {0.99, -0.99, -5.01}});
}

TEST_F(SolidConverterTest, para)
{
    this->build_and_test(
        G4Para("LArEMECInnerAluConeAluBar",
               5.01588152875291,
               5,
               514,
               0,
               4.56062963173385,
               0),
        R"json({"_type":"shape","interior":{"_type":"parallelepiped","alpha":0.0,"halfedges":[0.501588152875291,0.5,51.400000000000006],"phi":0.0,"theta":0.22584674950181247},"label":"LArEMECInnerAluConeAluBar"})json",
        {
            {51.2, 0.40, 7.76},
            {51.4, 0.51, 7.78},
        });
}

TEST_F(SolidConverterTest, polycone)
{
    {
        static double const z[] = {6, 630};
        static double const rmin[] = {0, 0};
        static double const rmax[] = {95, 95};
        this->build_and_test(
            G4Polycone("HGCalEE", 0, 360 * deg, std::size(z), z, rmin, rmax),
            R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"cone","halfheight":31.2,"radii":[9.5,9.5]},"label":"HGCalEE"},"transform":{"_type":"translation","data":[0.0,0.0,31.8]}})json",
            {{-6.72, -6.72, 0.7},
             {6.72, 6.72, 62.9},
             {0, 0, 31.8},
             {-9.5, -9.5, 0.5},
             {-6.72, 9.0, 0.70}});
    }
    {
        static double const z[] = {0, 5, 20, 20, 63.3, 115.2, 144};
        static double const rmin[] = {1954, 1954, 1954, 2016, 2016, 2044, 2044};
        static double const rmax[] = {2065, 2070, 2070, 2070, 2070, 2070, 2070};

        this->build_and_test(
            G4Polycone(
                "EMEC_FrontOuterRing", 0, 360 * deg, std::size(z), z, rmin, rmax),
            R"json({"_type":"polycone","label":"EMEC_FrontOuterRing","segments":[{"outer":[206.5,207.0,207.0,207.0,207.0,207.0,207.0],"z":[0.0,0.5,2.0,2.0,6.33,11.52,14.4]},["inner",[195.4,195.4,195.4,201.6,201.6,204.4,204.4]]]})json",
            {{0, 0, -0.1},
             {195.3, 0, 4.999},
             {195.5, 0, 4.999},
             {206.9, 0, 0.25},
             {204.5, 0, 14.3}});
    }
    {
        static double const z[] = {-165, -10, -10, 10, 10, 165};
        static double const rmin[] = {2044, 2044, 2050.5, 2050.5, 2044, 2044};
        static double const rmax[] = {2070, 2070, 2070, 2070, 2070, 2070};

        this->build_and_test(
            G4Polycone("EMEC_WideStretchers",
                       -5.625 * deg,
                       11.25 * deg,
                       std::size(z),
                       z,
                       rmin,
                       rmax),
            R"json({"_type":"polycone","enclosed_azi":{"stop":1.015625,"start":0.984375},"label":"EMEC_WideStretchers","segments":[{"outer":[207.0,207.0,207.0,207.0,207.0,207.0],"z":[-16.5,-1.0,-1.0,1.0,1.0,16.5]},["inner",[204.4,204.4,205.05,205.05,204.4,204.4]]]})json",
            {{206, 0, 0}, {-206, 0, 0}});
    }
}

TEST_F(SolidConverterTest, polyhedra)
{
    static double const z[] = {-0.6, 0.6};
    static double const rmin[] = {0, 0};
    static double const rmax[] = {61.85, 61.85};

    // Flat-top hexagon
    this->build_and_test(
        G4Polyhedra(
            "HGCalEEAbs", 330 * deg, 360 * deg, 6, std::size(z), z, rmin, rmax),
        R"json({"_type":"shape","interior":{"_type":"prism","apothem":6.1850000000000005,"halfheight":0.06,"num_sides":6,"orientation":0.5},"label":"HGCalEEAbs"})json",
        {{6.18, 6.18, 0.05},
         {0, 0, 0.06},
         {7.15, 7.15, 0.05},
         {3.0, 6.01, 0},
         {6.18, 7.15, 0}});

    // Triangle
    static double const z2[] = {10, 50};
    static double const rmin2[] = {0, 0};
    static double const rmax2[] = {10, 10};
    this->build_and_test(
        G4Polyhedra(
            "tri", 30 * deg, 360 * deg, 3, std::size(z2), z2, rmin2, rmax2),
        R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"prism","apothem":0.9999999999999999,"halfheight":2.0,"num_sides":3,"orientation":0.25},"label":"tri"},"transform":{"_type":"translation","data":[0.0,0.0,3.0]}})json",
        {
            {0, 0, 0.9},
            {0, 0, 1.1},
            {0, 0, 4.9},
            {0, 0, 5.1},
            {0, 1.01, 1.1},
            {0, -1.01, 1.1},
        });
    // Rotate 60 degrees
    this->build_and_test(
        G4Polyhedra(
            "tri", 60 * deg, 360 * deg, 3, std::size(z2), z2, rmin2, rmax2),
        R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"prism","apothem":0.9999999999999999,"halfheight":2.0,"num_sides":3,"orientation":0.5},"label":"tri"},"transform":{"_type":"translation","data":[0.0,0.0,3.0]}})json");
    // Rotate 90 degrees
    this->build_and_test(
        G4Polyhedra(
            "tri", 90 * deg, 360 * deg, 3, std::size(z2), z2, rmin2, rmax2),
        R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"prism","apothem":0.9999999999999999,"halfheight":2.0,"num_sides":3,"orientation":0.75},"label":"tri"},"transform":{"_type":"translation","data":[0.0,0.0,3.0]}})json");
}

TEST_F(SolidConverterTest, sphere)
{
    this->build_and_test(
        G4Sphere("Solid G4Sphere", 0, 50, 0, twopi, 0, pi),
        R"json({"_type":"shape","interior":{"_type":"sphere","radius":5.0},"label":"Solid G4Sphere"})json");
    this->build_and_test(
        G4Sphere("sn1", 0, 50, halfpi, 3. * halfpi, 0, pi),
        R"json({"_type":"solid","enclosed_azi":{"stop":1.0,"start":0.25},"interior":{"_type":"sphere","radius":5.0},"label":"sn1"})json",
        {{-3, 0.05, 0}, {3, 0.5, 0}, {-0.01, -0.01, 4.9}});
    this->build_and_test(
        G4Sphere("sn12", 0, 50, 0, twopi, 0., 0.25 * pi),
        R"json({"_type":"solid","enclosed_polar":{"start":0.0,"stop":0.125},"interior":{"_type":"sphere","radius":5.0},"label":"sn12"})json");

    this->build_and_test(
        G4Sphere("Spherical Shell", 45, 50, 0, twopi, 0, pi),
        R"json({"_type":"solid","excluded":{"_type":"sphere","radius":4.5},"interior":{"_type":"sphere","radius":5.0},"label":"Spherical Shell"})json",
        {{0, 0, 4.4}, {0, 0, 4.6}, {0, 0, 5.1}});
    this->build_and_test(
        G4Sphere("Band (theta segment1)", 45, 50, 0, twopi, pi * 3 / 4, pi / 4),
        R"json({"_type":"solid","enclosed_polar":{"start":0.375,"stop":0.5},"excluded":{"_type":"sphere","radius":4.5},"interior":{"_type":"sphere","radius":5.0},"label":"Band (theta segment1)"})json");

    this->build_and_test(
        G4Sphere("Band (phi segment)", 5, 50, -pi, 3. * pi / 2., 0, twopi),
        R"json({"_type":"solid","enclosed_azi":{"start":0.5,"stop":1.25},"excluded":{"_type":"sphere","radius":0.5},"interior":{"_type":"sphere","radius":5.0},"label":"Band (phi segment)"})json");
    this->build_and_test(
        G4Sphere(
            "Patch (phi/theta seg)", 45, 50, -pi / 4, halfpi, pi / 4, halfpi),
        R"json({"_type":"solid","enclosed_azi":{"start":0.875,"stop":1.125},"enclosed_polar":{"start":0.125,"stop":0.375},"excluded":{"_type":"sphere","radius":4.5},"interior":{"_type":"sphere","radius":5.0},"label":"Patch (phi/theta seg)"})json");

    this->build_and_test(
        G4Sphere("John example", 300, 500, 0, 5.76, 0, pi),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.9167324722093171,"start":0.0},"excluded":{"_type":"sphere","radius":30.0},"interior":{"_type":"sphere","radius":50.0},"label":"John example"})json");
}

TEST_F(SolidConverterTest, subtractionsolid)
{
    G4Tubs t1("Solid Tube #1", 0, 50., 50., 0., 360. * deg);
    G4Box b3("Test Box #3", 10., 20., 50.);
    G4RotationMatrix xRot;
    xRot.rotateZ(-pi);
    G4Transform3D const transform(xRot, G4ThreeVector(0, 30, 0));
    this->build_and_test(
        G4SubtractionSolid("t1Subtractionb3", &t1, &b3, transform),
        R"json({"_type":"all","daughters":[{"_type":"shape","interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Tube #1"},{"_type":"negated","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[1.0,2.0,5.0]},"label":"Test Box #3"},"transform":{"_type":"transformation","data":[-1.0,1.2246467991473532e-16,0.0,-1.2246467991473532e-16,-1.0,-0.0,0.0,0.0,1.0,0.0,3.0,0.0]}},"label":""}],"label":"t1Subtractionb3"})json",
        {{0, 0, 0}, {0, 0, 10}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
}

TEST_F(SolidConverterTest, reflectedsolid)
{
    // Triangle, flat top
    static double const z[] = {10, 50};
    static double const rmin[] = {0, 0};
    static double const rmax[] = {10, 10};
    G4Polyhedra tri("tri", 30 * deg, 360 * deg, 3, std::size(z), z, rmin, rmax);

    // Reflect across xy plane
    G4ReflectedSolid reflz("tri_refl", &tri, G4ScaleZ3D());
    this->build_and_test(
        reflz,
        R"json({"_type":"transformed","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"prism","apothem":0.9999999999999999,"halfheight":2.0,"num_sides":3,"orientation":0.25},"label":"tri"},"transform":{"_type":"translation","data":[0.0,0.0,3.0]}},"transform":{"_type":"transformation","data":[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]}})json",
        {
            {0, 0, 1.1},
            {0, 0, 5.1},
            {0, 0, -1.1},
            {0, 0, -5.1},
            {0, 1.0, -1.1},
        });

    // Reflect across yz plane
    G4ReflectedSolid reflx("tri_refl", &tri, G4ScaleX3D());
    this->build_and_test(
        reflx,
        R"json({"_type":"transformed","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"prism","apothem":0.9999999999999999,"halfheight":2.0,"num_sides":3,"orientation":0.25},"label":"tri"},"transform":{"_type":"translation","data":[0.0,0.0,3.0]}},"transform":{"_type":"transformation","data":[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]}})json",
        {
            {0, 0.99, 1.1},
            {0, -0.99, 5.1},
            {0, 1.01, 1.1},
            {0, -1.01, 5.1},
        });
}

TEST_F(SolidConverterTest, DISABLED_scaledsolid)
{
    G4Box b("box", 10., 20., 30.);
    G4ScaledSolid ss("scaled", &b, G4Scale3D(0.5, 1.0, 2.0));
    this->build_and_test(ss,
                         R"json(null)json",
                         {
                             {0.49, 0, 0},
                             {0.51, 0, 0},
                             {0.49, 0.99, 2.99},
                             {0.49, 0.99, 3.01},
                         });
}

TEST_F(SolidConverterTest, torus)
{
    G4Torus torus("testTorus", 0 * cm, 20 * cm, 50 * cm, 0 * deg, 270 * deg);
    auto json_str
        = R"json({"_type":"solid","enclosed_azi":{"stop":0.75,"start":0.0},"excluded":{"_type":"cylinder","halfheight":20.0,"radius":30.0},"interior":{"_type":"cylinder","halfheight":20.0,"radius":70.0},"label":"testTorus"})json";

    SolidConverter convert{scale_, transform_};
    auto obj = convert(torus);
    CELER_ASSERT(obj);
    EXPECT_JSON_EQ(json_str, to_string(*obj));
}

TEST_F(SolidConverterTest, trap)
{
    double tan_alpha = std::tan(45 * deg);
    this->build_and_test(
        G4Trap("trap0", 10, 0, 0, 10, 10, 10, tan_alpha, 10, 10, 10, tan_alpha),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":1.0,"lower":[[-0.5574077246549016,-1.0],[2.557407724654902,1.0],[0.5574077246549016,1.0],[-2.557407724654902,-1.0]],"upper":[[-0.5574077246549016,-1.0],[2.557407724654902,1.0],[0.5574077246549016,1.0],[-2.557407724654902,-1.0]]},"label":"trap0"})json");

    this->build_and_test(
        G4Trap("trap_box", 30, 0, 0, 20, 10, 10, 0, 20, 10, 10, 0),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":3.0,"lower":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]],"upper":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]]},"label":"trap_box"})json",
        {{-1, -2, -3}, {1, 2, 3}, {1, 2, 4}});

    this->build_and_test(
        G4Trap("trap_trd", 50, 100, 100, 200, 300),
        R"json({"_type":"shape","interior":{
"_type":"genprism",
"halfheight":30.0,
"lower":[[5.0,-10.0],[5.0,10.0],[-5.0,10.0],[-5.0,-10.0]],
"upper":[[10.0,-20.0],[10.0,20.0],[-10.0,20.0],[-10.0,-20.0]]
},"label":"trap_trd"})json",
        {{-10, -20, -40}, {-10, -20, -30 + 1.e-6}, {5, 10, 30}, {10, 10, 30}});

    tan_alpha = std::tan(15 * deg);
    this->build_and_test(
        G4Trap("trap1",
               40,
               5 * deg,
               10 * deg,
               20,
               10,
               10,
               tan_alpha,
               30,
               15,
               15,
               tan_alpha),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":4.0,"lower":[[0.10625895161288212,-2.060768987951168],[1.2044649352590673,1.9392310120488323],[-0.7955350647409326,1.9392310120488323],[-1.8937410483871178,-2.060768987951168]],"upper":[[1.0209835688293862,-2.939231012048832],[2.668292544298664,3.060768987951168],[-0.33170745570133575,3.060768987951168],[-1.9790164311706138,-2.939231012048832]]},"label":"trap1"})json",
        {{-1.89, -2.1, -4.01},
         {0.12, -2.07, -4.01},
         {1.20, 1.94, -4.01},
         {-0.81, 1.9, -4.01},
         {-1.96, -2.94, 4.01}});

    tan_alpha = std::tan(30 * deg);
    this->build_and_test(
        G4Trap("trap2",
               10,
               10 * deg,
               -15 * deg,
               20,
               10,
               10,
               tan_alpha,
               30,
               15,
               15,
               tan_alpha),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":1.0,"lower":[[-0.4730945579141699,-1.9543632192272244],[2.132456988838409,2.0456367807727753],[0.13245698883840862,2.0456367807727753],[-2.4730945579141697,-1.9543632192272244]],"upper":[[-0.28384487552655324,-3.0456367807727753],[3.6244824446023145,2.9543632192272247],[0.6244824446023145,2.9543632192272247],[-3.2838448755265532,-3.0456367807727753]]},"label":"trap2"})json",
        {{-2.33, -1.96, -1.01},
         {-2.32, -1.95, -0.99},
         {3.41, 2.96, 1.01},
         {3.40, 2.95, 0.99}});

    this->build_and_test(
        G4Trap(/* name = */ "TileTB_FingerIron",
               /* z = */ 362 * half,
               /* theta = */ 0 * deg,
               /* phi = */ 0 * deg,
               /* y1 = */ 360 * half,
               /* x1 = */ 40 * half,
               /* x2 = */ 22.5 * half,
               /* alpha1 = */ -1.39233161727723 * deg,
               /* y2 = */ 360 * half,
               /* x3 = */ 40 * half,
               /* x4 = */ 22.5 * half,
               /* alpha2 = */ -1.39233161727723 * deg),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":18.1,"lower":[[2.4375000000000013,-18.0],[0.6874999999999987,18.0],[-1.5625000000000013,18.0],[-1.5624999999999987,-18.0]],"upper":[[2.4375000000000013,-18.0],[0.6874999999999987,18.0],[-1.5625000000000013,18.0],[-1.5624999999999987,-18.0]]},"label":"TileTB_FingerIron"})json");

    this->build_and_test(
        G4Trap(/* name = */ "cms_hllhc_notch_ext",
               /* z = */ 126.5 * half,
               /* theta = */ 32.7924191 * deg,
               /* phi = */ 0 * deg,
               /* y1 = */ 465 * half,
               /* x1 = */ 200 * half,
               /* x2 = */ 200 * half,
               /* alpha1 = */ 0 * deg,
               /* y2 = */ 350 * half,
               /* x3 = */ 281.5 * half,
               /* x4 = */ 281.5 * half,
               /* alpha2 = */ 0 * deg),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":6.325,"lower":[[5.92499999773904,-23.25],[5.92499999773904,23.25],[-14.07500000226096,23.25],[-14.07500000226096,-23.25]],"upper":[[18.15000000226096,-17.5],[18.15000000226096,17.5],[-9.999999997739042,17.5],[-9.999999997739042,-17.5]]},"label":"cms_hllhc_notch_ext"})json");
}

TEST_F(SolidConverterTest, trd)
{
    this->build_and_test(
        G4Trd("trd_box", 10, 10, 20, 20, 30),
        R"json({"_type":"shape","interior":{"_type":"genprism","halfheight":3.0,"lower":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]],"upper":[[1.0,-2.0],[1.0,2.0],[-1.0,2.0],[-1.0,-2.0]]},"label":"trd_box"})json",
        {{-1, -2, -3}, {1, 2, 3}, {1, 2, 4}});

    this->build_and_test(
        G4Trd("trd", 50, 100, 100, 200, 300),
        R"json({
"_type":"shape",
"interior":{"_type":"genprism","halfheight":30.0,
"lower":[[5.0,-10.0],[5.0,10.0],[-5.0,10.0],[-5.0,-10.0]],
"upper":[[10.0,-20.0],[10.0,20.0],[-10.0,20.0],[-10.0,-20.0]]},
"label":"trd"
})json",
        {{-10, -20, -40}, {-10, -20, -30 + 1.e-6}, {5, 10, 30}, {10, 10, 30}});
}

TEST_F(SolidConverterTest, tubs)
{
    this->build_and_test(
        G4Tubs("Solid Tube #1", 0, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"shape","interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Tube #1"})json");

    this->build_and_test(
        G4Tubs("Solid Tube #1a", 0, 50 * mm, 50 * mm, 0, 0.5 * pi),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.25,"start":0.0},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Tube #1a"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2", 45 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":4.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2a", 5 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":0.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2a"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2b", 15 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":1.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2b"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2c", 25 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":2.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2c"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2d", 35 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":3.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2d"})json");

    this->build_and_test(
        G4Tubs("Solid Sector #3", 0, 50 * mm, 50 * mm, halfpi, halfpi),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.5,"start":0.25},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Sector #3"})json");

    this->build_and_test(
        G4Tubs("Hole Sector #4", 45 * mm, 50 * mm, 50 * mm, halfpi, halfpi),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.5,"start":0.25},"excluded":{"_type":"cylinder","halfheight":5.0,"radius":4.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Sector #4"})json");

    this->build_and_test(
        G4Tubs("Hole Sector #5", 50 * mm, 100 * mm, 50 * mm, 0.0, 270.0 * deg),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.75,"start":0.0},"excluded":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"interior":{"_type":"cylinder","halfheight":5.0,"radius":10.0},"label":"Hole Sector #5"})json");

    this->build_and_test(
        G4Tubs("Solid Sector #3", 0, 50 * mm, 50 * mm, halfpi, 3. * halfpi),
        R"json({"_type":"solid","enclosed_azi":{"stop":1.0,"start":0.25},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Sector #3"})json");

    this->build_and_test(
        G4Tubs("Barrel",
               2288 * mm,
               4250 * mm,
               (5640.0 / 2) * mm,
               0 * deg,
               11.25 * deg),
        R"json({"_type":"solid","enclosed_azi":{"stop":0.03125,"start":0.0},"excluded":{"_type":"cylinder","halfheight":282.0,"radius":228.8},"interior":{"_type":"cylinder","halfheight":282.0,"radius":425.0},"label":"Barrel"})json",
        {{300, 25, 0.1}, {300, -25, 0.1}, {450, 0.1, 0.1}});
}

TEST_F(SolidConverterTest, unionsolid)
{
    G4Tubs t1("Solid Tube #1", 0, 50, 50, 0, 360 * deg);
    G4Box b3("Test Box #3", 10, 20, 50);
    G4RotationMatrix identity, xRot;
    xRot.rotateZ(-pi);
    G4Transform3D transform(xRot, G4ThreeVector(0, 40, 0));

    this->build_and_test(
        G4UnionSolid("t1Unionb3", &t1, &b3, transform),
        R"json({"_type":"any","daughters":[{"_type":"shape","interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Tube #1"},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[1.0,2.0,5.0]},"label":"Test Box #3"},"transform":{"_type":"transformation","data":[-1.0,1.2246467991473532e-16,0.0,-1.2246467991473532e-16,-1.0,-0.0,0.0,0.0,1.0,0.0,4.0,0.0]}}],"label":"t1Unionb3"})json",
        {
            {0, 6, 0},
            {5, 0, 0},
            {0, 6.5, 0},
            {0, 4.9, 0},
            {0, 5.1, 0},
        });
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
