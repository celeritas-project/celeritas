//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/SolidConverter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/SolidConverter.hh"

#include <initializer_list>
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
#include <G4Navigator.hh>
#include <G4Orb.hh>
#include <G4PVDivision.hh>
#include <G4PVParameterised.hh>
#include <G4PVReplica.hh>
#include <G4Para.hh>
#include <G4Paraboloid.hh>
#include <G4PhysicalConstants.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4ReflectedSolid.hh>
#include <G4RotationMatrix.hh>
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

//---------------------------------------------------------------------------//
class SolidConverterTest : public ::celeritas::orangeinp::test::ObjectTestBase
{
  protected:
    Tol tolerance() const { return Tol::from_default(); }

    void build_and_test(G4VSolid const& solid,
                        std::string_view json_str = "null",
                        std::initializer_list<Real3> points = {})
    {
        SCOPED_TRACE(solid.GetName());
        // Convert the object
        auto obj = convert_(solid);
        CELER_ASSERT(obj);
        EXPECT_JSON_EQ(json_str, to_string(*obj));

        // Construct a volume from it
        auto vol_id = this->build_volume(*obj);

        // Set up functions to calculate in/on/out
        auto const& u = this->unit();
        CELER_ASSERT(vol_id < u.volumes.size());
        auto calc_org_sense
            = [&u, node = u.volumes[vol_id.get()]](Real3 const& pos) {
                  SenseEvaluator eval_sense(u.tree, u.surfaces, pos);
                  return eval_sense(node);
              };
        auto calc_g4_sense = [&solid,
                              inv_scale = 1 / scale_(1.0)](Real3 const& pos) {
            return to_signed_sense(solid.Inside(G4ThreeVector(
                inv_scale * pos[0], inv_scale * pos[1], inv_scale * pos[2])));
        };

        for (Real3 const& r : points)
        {
            EXPECT_EQ(calc_g4_sense(r), calc_org_sense(r))
                << "at " << r << " [cm]";
        }
    }

    Scaler scale_{0.1};
    Transformer transform_{scale_};
    SolidConverter convert_{scale_, transform_};
};

TEST_F(SolidConverterTest, box)
{
    this->build_and_test(
        G4Box("Test Box", 20, 30, 40),
        R"json({"_type":"shape","interior":{"_type":"box","halfwidths":[2.0,3.0,4.0]},"label":"Test Box"})json",
        {{10, 0, 0}, {0, 30, 0}, {0, 0, 41}});
}

TEST_F(SolidConverterTest, cons)
{
    this->build_and_test(
        G4Cons("Solid TubeLike #1", 0, 50, 0, 50, 50, 0, 360),
        R"json({"_type":"shape","interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid TubeLike #1"})json",
        {{0, 0, 4}, {0, 0, 5}, {0, 0, 6}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0}});

    this->print_expected();

    this->build_and_test(
        G4Cons("test10",
               20.0,
               80.0,
               60.0,
               140.0,
               1.0,
               0.17453292519943,
               5.235987755983),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.8333333333333353,"start":0.027777777777777308},"excluded":{"_type":"cone","halfheight":0.1,"radii":[2.0,6.0]},"interior":{"_type":"cone","halfheight":0.1,"radii":[8.0,14.0]},"label":"test10"})json");

    this->build_and_test(
        G4Cons(
            "aCone", 2 * cm, 6 * cm, 8 * cm, 14 * cm, 10 * cm, 10 * deg, 300 * deg),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.8333333333333334,"start":0.027777777777777776},"excluded":{"_type":"cone","halfheight":10.0,"radii":[2.0,8.0]},"interior":{"_type":"cone","halfheight":10.0,"radii":[6.0,14.0]},"label":"aCone"})json");
}

TEST_F(SolidConverterTest, displaced)
{
    G4Box b1("Test Box #1", 20, 30, 40);
    G4Box b2("Test Box #2", 10, 10, 10);
    G4Tubs t3("Solid cutted Tube #3", 0, 50, 50, 0, pi / 2.0);

    G4ThreeVector ponb2mx(-10, 0, 0), pzero(0, 0, 0);
    G4RotationMatrix xRot;
    xRot.rotateZ(-pi * 0.5);
    G4Transform3D transform(xRot, pzero);

    this->build_and_test(
        G4DisplacedSolid("passRotT3", &t3, &xRot, ponb2mx),
        R"json({"_type":"transformed","daughter":{"_type":"solid","enclosed_angle":{"interior":0.25,"start":0.0},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid cutted Tube #3"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,-1.0,0.0,1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,-1.0,0.0,0.0]}})json");
    this->build_and_test(
        G4DisplacedSolid("actiRotT3", &t3, transform),
        R"json({"_type":"transformed","daughter":{"_type":"solid","enclosed_angle":{"interior":0.25,"start":0.0},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid cutted Tube #3"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,1.0,0.0,-1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,0.0,0.0,0.0]}})json");
    this->build_and_test(
        G4DisplacedSolid("actiRotB3", &b1, transform),
        R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[2.0,3.0,4.0]},"label":"Test Box #1"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,1.0,0.0,-1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,0.0,0.0,0.0]}})json");
    this->build_and_test(
        G4DisplacedSolid("passRotT3", &b2, &xRot, ponb2mx),
        R"json({"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[1.0,1.0,1.0]},"label":"Test Box #2"},"transform":{"_type":"transformation","data":[6.123233995736766e-17,-1.0,0.0,1.0,6.123233995736766e-17,0.0,0.0,0.0,1.0,-1.0,0.0,0.0]}})json");
}

TEST_F(SolidConverterTest, intersectionsolid) {}

TEST_F(SolidConverterTest, orb) {}

TEST_F(SolidConverterTest, polycone) {}

TEST_F(SolidConverterTest, polyhedra) {}

TEST_F(SolidConverterTest, subtractionsolid) {}

TEST_F(SolidConverterTest, tubs)
{
    this->build_and_test(
        G4Tubs("Solid Tube #1", 0, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"shape","interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Tube #1"})json");

    this->build_and_test(
        G4Tubs("Solid Tube #1", 0, 50 * mm, 50 * mm, 0, 0.5 * pi),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.25,"start":0.0},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Tube #1"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2", 45 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":4.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2", 5 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":0.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2", 15 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":1.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2", 25 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":2.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2"})json");

    this->build_and_test(
        G4Tubs("Hole Tube #2", 35 * mm, 50 * mm, 50 * mm, 0, 2 * pi),
        R"json({"_type":"solid","excluded":{"_type":"cylinder","halfheight":5.0,"radius":3.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Tube #2"})json");

    this->build_and_test(
        G4Tubs("Solid Sector #3", 0, 50 * mm, 50 * mm, halfpi, halfpi),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.25,"start":0.25},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Sector #3"})json");

    this->build_and_test(
        G4Tubs("Hole Sector #4", 45 * mm, 50 * mm, 50 * mm, halfpi, halfpi),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.25,"start":0.25},"excluded":{"_type":"cylinder","halfheight":5.0,"radius":4.5},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Hole Sector #4"})json");

    this->build_and_test(
        G4Tubs("Hole Sector #5", 50 * mm, 100 * mm, 50 * mm, 0.0, 270.0 * deg),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.75,"start":0.0},"excluded":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"interior":{"_type":"cylinder","halfheight":5.0,"radius":10.0},"label":"Hole Sector #5"})json");

    this->build_and_test(
        G4Tubs("Solid Sector #3", 0, 50 * mm, 50 * mm, halfpi, 3. * halfpi),
        R"json({"_type":"solid","enclosed_angle":{"interior":0.75,"start":0.25},"interior":{"_type":"cylinder","halfheight":5.0,"radius":5.0},"label":"Solid Sector #3"})json");
}

TEST_F(SolidConverterTest, unionsolid) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
