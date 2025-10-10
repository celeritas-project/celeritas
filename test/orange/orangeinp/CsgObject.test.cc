//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgObject.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/CsgObject.hh"

#include "orange/MatrixUtils.hh"
#include "orange/orangeinp/Shape.hh"
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

class CsgObjectTest : public ObjectTestBase
{
  protected:
    using SPConstObject = std::shared_ptr<ObjectInterface const>;
    using VecObject = std::vector<SPConstObject>;

    Tol tolerance() const override { return Tol::from_relative(1e-4); }

    //! Avoid implicit floating point casts by calling this function
    SPConstObject make_sphere(std::string&& name, real_type radius)
    {
        return make_shape<Sphere>(std::move(name), radius);
    }

    //! Create a translated object
    SPConstObject
    make_translated(SPConstObject const& shape, Real3 const& translation)
    {
        return std::make_shared<Transformed>(shape, Translation{translation});
    }
};

//---------------------------------------------------------------------------//
// NEGATED_OBJECT
//---------------------------------------------------------------------------//

using NegatedObjectTest = CsgObjectTest;

TEST_F(NegatedObjectTest, just_neg)
{
    auto sph = this->make_sphere("sph", 1);
    auto trsph = this->make_translated(sph, {0, 0, 1});

    this->build_volume(NegatedObject{"antisph", sph});
    this->build_volume(NegatedObject{"antitrsph", trsph});

    static char const* const expected_volume_strings[] = {"+0", "+1"};
    static char const* const expected_md_strings[]
        = {"", "", "antisph,sph@s", "sph", "antitrsph,sph@s", "sph"};
    static char const* const expected_bound_strings[] = {
        R"(~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(~4: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, {1,1,2}}})",
        R"(5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, {1,1,2}}})",
    };
    static char const* const expected_trans_strings[]
        = {"2: t=0 -> {}", "3: t=0", "4: t=0", "5: t=1 -> {{0,0,1}}"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
}

TEST_F(NegatedObjectTest, pos_neg)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto trsph = this->make_translated(sph, {0, 0, 1});
    auto antitrsph = std::make_shared<NegatedObject>("antitrsph", trsph);

    this->build_volume(*sph);
    this->build_volume(NegatedObject{"antisph", sph});
    this->build_volume(*trsph);
    this->build_volume(*antitrsph);

    static char const* const expected_volume_strings[]
        = {"-0", "+0", "-1", "+1"};
    static char const* const expected_md_strings[]
        = {"", "", "antisph,sph@s", "sph", "antitrsph,sph@s", "sph"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_JSON_EQ(
        R"json({"_type":"negated","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"sphere","radius":1.0},"label":"sph"},"transform":{"_type":"translation","data":[0.0,0.0,1.0]}},"label":"antitrsph"})json",
        to_string(*antitrsph));
}

TEST_F(NegatedObjectTest, double_neg)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto antisph = std::make_shared<NegatedObject>("antisph", sph);

    this->build_volume(NegatedObject{"antiantisph", antisph});

    static char const* const expected_volume_strings[] = {"-0"};
    static char const* const expected_md_strings[]
        = {"", "", "antisph,sph@s", "antiantisph,sph"};
    static char const* const expected_bound_strings[] = {
        R"(~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
    };
    static int const expected_volume_nodes[] = {3};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
}

//---------------------------------------------------------------------------//
// ANY_OBJECTS
//---------------------------------------------------------------------------//

using AnyObjectsTest = CsgObjectTest;

TEST_F(AnyObjectsTest, adjoining)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto trsph = this->make_translated(sph, {0, 0, 1});

    AnyObjects const anysph{"anysph", {sph, trsph}};
    this->build_volume(anysph);

    static char const* const expected_surface_strings[]
        = {"Sphere: r=1", "Sphere: r=1 at {0,0,1}"};
    static char const* const expected_volume_strings[] = {"any(-0, -1)"};
    static char const* const expected_md_strings[]
        = {"", "", "sph@s", "sph", "sph@s", "sph", "anysph"};
    static char const* const expected_bound_strings[] = {
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, {1,1,2}}})",
        R"(6: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,-1}, {1,1,2}}})",
    };
    static char const* const expected_trans_strings[]
        = {"3: t=0 -> {}", "5: t=1 -> {{0,0,1}}", "6: t=0"};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["~",2],["S",1],["~",4],["|",[3,5]]])json";

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    EXPECT_JSON_EQ(
        R"json({"_type":"any","daughters":[{"_type":"shape","interior":{"_type":"sphere","radius":1.0},"label":"sph"},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"sphere","radius":1.0},"label":"sph"},"transform":{"_type":"translation","data":[0.0,0.0,1.0]}}],"label":"anysph"})json",
        to_string(anysph));
}

//---------------------------------------------------------------------------//
// ALL_OBJECTS
//---------------------------------------------------------------------------//

using AllObjectsTest = CsgObjectTest;

TEST_F(AllObjectsTest, overlapping)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto trsph = this->make_translated(sph, {0, 0, 1});

    this->build_volume(AllObjects{"allsph", {sph, trsph}});

    static char const* const expected_volume_strings[] = {"all(-0, -1)"};
    static char const* const expected_md_strings[]
        = {"", "", "sph@s", "sph", "sph@s", "sph", "allsph"};
    static char const* const expected_bound_strings[] = {
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, {1,1,2}}})",
        R"(6: {{{-0.866,-0.866,0.134}, {0.866,0.866,0.866}}, {{-1,-1,0}, {1,1,1}}})",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(AllObjectsTest, identical)
{
    auto sph = this->make_sphere("sph", 1.0);

    this->build_volume(AllObjects{"allsph", {sph, sph}});

    static char const* const expected_volume_strings[] = {"-0"};
    static char const* const expected_md_strings[]
        = {"", "", "sph@s", "allsph,sph"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(AllObjectsTest, disjoint)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto trsph = this->make_translated(sph, {0, 0, 2.5});

    this->build_volume(AllObjects{"allsph", {sph, trsph}});

    static char const* const expected_volume_strings[] = {"all(-0, -1)"};
    static char const* const expected_bound_strings[] = {
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(5: {{{-0.866,-0.866,1.63}, {0.866,0.866,3.37}}, {{-1,-1,1.5}, {1,1,3.5}}})",
        "6: {null, null}",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(AllObjectsTest, allneg)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto trsph = this->make_translated(sph, {0, 0, 1});
    auto trsph2 = this->make_translated(sph, {0, 0, 2});

    AllObjects const allsph{"allsph",
                            {std::make_shared<NegatedObject>(sph),
                             std::make_shared<NegatedObject>(trsph),
                             std::make_shared<NegatedObject>(trsph2)}};
    this->build_volume(allsph);

    static char const* const expected_volume_strings[] = {"all(+0, +1, +2)"};
    static char const* const expected_md_strings[]
        = {"", "", "sph@s", "sph", "sph@s", "sph", "sph@s", "sph", "allsph"};
    static char const* const expected_bound_strings[] = {
        R"(~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(~4: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, {1,1,2}}})",
        R"(5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, {1,1,2}}})",
        R"(~6: {{{-0.866,-0.866,1.13}, {0.866,0.866,2.87}}, {{-1,-1,1}, {1,1,3}}})",
        R"(7: {{{-0.866,-0.866,1.13}, {0.866,0.866,2.87}}, {{-1,-1,1}, {1,1,3}}})",
        "8: {null, inf}",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_JSON_EQ(
        R"json({"_type":"all","daughters":[{"_type":"negated","daughter":{"_type":"shape","interior":{"_type":"sphere","radius":1.0},"label":"sph"},"label":""},{"_type":"negated","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"sphere","radius":1.0},"label":"sph"},"transform":{"_type":"translation","data":[0.0,0.0,1.0]}},"label":""},{"_type":"negated","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"sphere","radius":1.0},"label":"sph"},"transform":{"_type":"translation","data":[0.0,0.0,2.0]}},"label":""}],"label":"allsph"})json",
        to_string(allsph));
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

TEST_F(CsgObjectTest, subtraction)
{
    auto apple = this->make_sphere("apple", 1.0);
    auto bite
        = this->make_translated(this->make_sphere("bite", 0.5), {0, 0, 1});

    auto sub = make_subtraction("nomnom", apple, bite);
    ASSERT_TRUE(sub);
    this->build_volume(*sub);
    static char const* const expected_volume_strings[] = {"all(-0, +1)"};
    static char const* const expected_md_strings[]
        = {"", "", "apple@s", "apple", "bite@s", "bite", "nomnom"};
    static char const* const expected_bound_strings[] = {
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(~4: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, {{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}})",
        R"(5: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, {{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}})",
        "6: {null, inf}",
    };
    static char const* const expected_trans_strings[]
        = {"3: t=0 -> {}", "4: t=0", "5: t=1 -> {{0,0,1}}", "6: t=0"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
}

// Add a test of an object that can't be deleted from the ATLAS boundary
TEST_F(CsgObjectTest, subtraction_atlas)
{
    // Shape definitions are from SolidConverter.test.cc
    using VR2 = GenPrism::VecReal2;
    auto trap = make_shape<GenPrism>(
        "trap",
        23.75,
        VR2{{4.75, -30.70}, {4.75, 30.70}, {-4.75, 30.70}, {-4.75, -30.70}},
        VR2{{4.75, -25.917}, {4.75, 25.917}, {-4.75, 25.917}, {-4.75, -25.917}});
    auto box = make_shape<Box>("box", Real3{5.0, 24.48, 15.0});
    auto trbox = std::make_shared<Transformed>(
        box,
        Transformation{make_rotation(Axis::x, Turn{41.592 / 360}),
                       {0, -22.349, 19.388}});

    auto sub = make_subtraction("LAr::DM::SPliceBoxr", trap, trbox);
    ASSERT_TRUE(sub);
    EXPECT_JSON_EQ(
        R"json({"_type":"all","daughters":[{"_type":"shape","interior":{"_type":"genprism","halfheight":23.75,"lower":[[4.75,-30.7],[4.75,30.7],[-4.75,30.7],[-4.75,-30.7]],"upper":[[4.75,-25.917],[4.75,25.917],[-4.75,25.917],[-4.75,-25.917]]},"label":"trap"},{"_type":"negated","daughter":{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"box","halfwidths":[5.0,24.48,15.0]},"label":"box"},"transform":{"_type":"transformation","data":[1.0,0.0,0.0,0.0,0.7478907847960848,-0.6638217938702345,0.0,0.6638217938702345,0.7478907847960848,0.0,-22.349,19.388]}},"label":""}],"label":"LAr::DM::SPliceBoxr"})json",
        to_string(*sub));

    this->build_volume(*sub);

    static char const* const expected_surface_strings[] = {
        "Plane: z=-23.75",
        "Plane: z=23.75",
        "Plane: x=4.75",
        "Plane: n={0,0.99497,0.10019}, d=28.166",
        "Plane: x=-4.75",
        "Plane: n={0,0.99497,-0.10019}, d=-28.166",
        "Plane: x=-5",
        "Plane: x=5",
        "Plane: n={0,0.74789,0.66382}, d=-28.324",
        "Plane: n={0,0.74789,0.66382}, d=20.636",
        "Plane: n={0,0.66382,-0.74789}, d=-14.336",
        "Plane: n={0,0.66382,-0.74789}, d=-44.336",
    };
    static char const* const expected_volume_strings[]
        = {"all(+0, -1, -2, -3, +4, +5, !all(+6, -7, +8, -9, -10, +11))"};
    static char const* const expected_md_strings[] = {
        "",       "",        "trap@mz", "trap@pz",
        "",       "trap@p0", "",        "trap@p1",
        "",       "trap@p2", "trap@p3", "trap",
        "box@mx", "box@px",  "",        "box@my",
        "box@py", "",        "box@mz",  "",
        "box@pz", "box",     "",        "LAr::DM::SPliceBoxr",
    };
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["S",1],["~",3],["S",2],["~",5],["S",3],["~",7],["S",4],["S",5],["&",[2,4,6,8,9,10]],["S",6],["S",7],["~",13],["S",8],["S",9],["~",16],["S",10],["~",18],["S",11],["&",[12,14,15,17,19,20]],["~",21],["&",[2,4,6,8,9,10,22]]])json";

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
}

TEST_F(CsgObjectTest, rdv)
{
    auto apple = this->make_sphere("apple", 1.0);
    auto bite
        = this->make_translated(this->make_sphere("bite", 0.5), {0, 0, 1});
    auto apple2
        = this->make_translated(this->make_sphere("apple2", 1.25), {0, 0, 4});

    this->build_volume(
        *make_rdv("bitten", {{Sense::inside, apple}, {Sense::outside, bite}}));
    // XXX low-level transform conflicts with lack of transform for this RDV
    this->build_volume(*make_rdv("forgotten", {{Sense::inside, apple2}}));
    this->build_volume(
        *make_rdv("air", {{Sense::outside, apple}, {Sense::outside, apple2}}));
    this->build_volume(
        *make_rdv("biteair", {{Sense::inside, apple}, {Sense::inside, bite}}));

    static char const* const expected_volume_strings[]
        = {"all(-0, +1)", "-2", "all(+0, +2)", "all(-0, -1)"};
    static char const* const expected_md_strings[] = {
        "",
        "",
        "apple@s",
        "apple",
        "bite@s",
        "bite",
        "bitten",
        "apple2@s",
        "apple2,forgotten",
        "air",
        "biteair",
    };
    static char const* const expected_bound_strings[] = {
        R"(~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, {1,1,1}}})",
        R"(~4: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, {{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}})",
        R"(5: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, {{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}})",
        "6: {null, inf}",
        R"(~7: {{{-1.08,-1.08,2.92}, {1.08,1.08,5.08}}, {{-1.25,-1.25,2.75}, {1.25,1.25,5.25}}})",
        R"(8: {{{-1.08,-1.08,2.92}, {1.08,1.08,5.08}}, {{-1.25,-1.25,2.75}, {1.25,1.25,5.25}}})",
        "9: {null, inf}",
        R"(10: {{{-0.433,-0.433,0.567}, {0.433,0.433,0.866}}, {{-0.5,-0.5,0.5}, {0.5,0.5,1}}})",
    };
    static char const* const expected_trans_strings[] = {
        "2: t=0 -> {}",
        "3: t=0",
        "4: t=0",
        "5: t=1 -> {{0,0,1}}",
        "6: t=0",
        "7: t=0",
        "8: t=2 -> {{0,0,4}}",
        "9: t=0",
        "10: t=0",
    };
    static int const expected_volume_nodes[] = {6, 8, 9, 10};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
}

TEST_F(CsgObjectTest, output)
{
    auto box = std::make_shared<BoxShape>("box", Box{{1, 1, 2}});
    auto cone = std::make_shared<ConeShape>("cone", Cone{{1.0, 0.5}, 2.0});
    auto cyl = std::make_shared<CylinderShape>("cyl", Cylinder{1.0, 2.0});
    auto ell = std::make_shared<EllipsoidShape>("ell", Ellipsoid{{1, 2, 3}});
    auto pri = std::make_shared<PrismShape>("rhex", Prism{6, 1.0, 2.0, 0.5});
    auto sph = std::make_shared<SphereShape>("sph", Sphere{1.25});

    auto trcyl = std::make_shared<Transformed>(
        cyl, Transformation{make_rotation(Axis::x, Turn{0.125}), {1, 2, 3}});
    auto trsph = std::make_shared<Transformed>(sph, Translation{{1, 2, 3}});

    auto all = std::make_shared<AllObjects>(
        "all_quadric", VecObject{cone, trcyl, ell, trsph});
    auto any = std::make_shared<AnyObjects>("any_planar", VecObject{box, pri});
    auto negany = std::make_shared<NegatedObject>("none_planar", any);

    EXPECT_JSON_EQ(
        R"json({"_type":"all","daughters":[{"_type":"shape","interior":{"_type":"cone","halfheight":2.0,"radii":[1.0,0.5]},"label":"cone"},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"cylinder","halfheight":2.0,"radius":1.0},"label":"cyl"},"transform":{"_type":"transformation","data":[1.0,0.0,0.0,0.0,0.7071067811865475,-0.7071067811865475,0.0,0.7071067811865475,0.7071067811865475,1.0,2.0,3.0]}},{"_type":"shape","interior":{"_type":"ellipsoid","radii":[1.0,2.0,3.0]},"label":"ell"},{"_type":"transformed","daughter":{"_type":"shape","interior":{"_type":"sphere","radius":1.25},"label":"sph"},"transform":{"_type":"translation","data":[1.0,2.0,3.0]}}],"label":"all_quadric"})json",
        to_string(*all));
    EXPECT_JSON_EQ(
        R"json({"_type":"negated","daughter":{"_type":"any","daughters":[{"_type":"shape","interior":{"_type":"box","halfwidths":[1.0,1.0,2.0]},"label":"box"},{"_type":"shape","interior":{"_type":"prism","apothem":1.0,"halfheight":2.0,"num_sides":6,"orientation":0.5},"label":"rhex"}],"label":"any_planar"},"label":"none_planar"})json",
        to_string(*negany));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
