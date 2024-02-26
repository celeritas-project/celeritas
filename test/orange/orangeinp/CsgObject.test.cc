//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgObject.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/CsgObject.hh"

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

    Tol tolerance() const override { return Tol::from_relative(1e-4); }

    //! Avoid implicit floating point casts by calling this function
    SPConstObject make_sphere(std::string&& name, real_type radius)
    {
        return std::make_shared<SphereShape>(std::move(name), radius);
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
        "~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
        "{1,1,1}}}",
        "3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
        "{1,1,1}}}",
        "~4: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, "
        "{1,1,2}}}",
        "5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, "
        "{1,1,2}}}",
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

    this->build_volume(*sph);
    this->build_volume(NegatedObject{"antisph", sph});
    this->build_volume(*trsph);
    this->build_volume(NegatedObject{"antitrsph", trsph});

    static char const* const expected_volume_strings[]
        = {"-0", "+0", "-1", "+1"};
    static char const* const expected_md_strings[]
        = {"", "", "antisph,sph@s", "sph", "antitrsph,sph@s", "sph"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(NegatedObjectTest, double_neg)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto antisph = std::make_shared<NegatedObject>("antisph", sph);

    this->build_volume(NegatedObject{"antiantisph", antisph});

    static char const* const expected_volume_strings[] = {"-0"};
    static char const* const expected_md_strings[]
        = {"", "", "antisph,sph@s", "antiantisph,sph"};
    static char const* const expected_bound_strings[]
        = {"~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}"};
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

    this->build_volume(AnyObjects{"anysph", {sph, trsph}});

    static char const* const expected_surface_strings[]
        = {"Sphere: r=1", "Sphere: r=1 at {0,0,1}"};
    static char const* const expected_volume_strings[] = {"any(-0, -1)"};
    static char const* const expected_md_strings[]
        = {"", "", "sph@s", "sph", "sph@s", "sph", "anysph"};
    static char const* const expected_bound_strings[] = {
        "3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
        "{1,1,1}}}",
        "5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, "
        "{1,1,2}}}",
        "6: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,-1}, "
        "{1,1,2}}}",
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
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
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
    static char const* const expected_bound_strings[]
        = {"3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, "
           "{1,1,2}}}",
           "6: {{{-0.866,-0.866,0.134}, {0.866,0.866,0.866}}, {{-1,-1,0}, "
           "{1,1,1}}}"};

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
    static char const* const expected_bound_strings[]
        = {"3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "5: {{{-0.866,-0.866,1.63}, {0.866,0.866,3.37}}, {{-1,-1,1.5}, "
           "{1,1,3.5}}}",
           "6: {null, null}"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(AllObjectsTest, allneg)
{
    auto sph = this->make_sphere("sph", 1.0);
    auto trsph = this->make_translated(sph, {0, 0, 1});
    auto trsph2 = this->make_translated(sph, {0, 0, 2});

    this->build_volume(AllObjects{"allsph",
                                  {std::make_shared<NegatedObject>(sph),
                                   std::make_shared<NegatedObject>(trsph),
                                   std::make_shared<NegatedObject>(trsph2)}});

    static char const* const expected_volume_strings[] = {"all(+0, +1, +2)"};
    static char const* const expected_md_strings[]
        = {"", "", "sph@s", "sph", "sph@s", "sph", "sph@s", "sph", "allsph"};
    static char const* const expected_bound_strings[]
        = {"~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "~4: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, "
           "{1,1,2}}}",
           "5: {{{-0.866,-0.866,0.134}, {0.866,0.866,1.87}}, {{-1,-1,0}, "
           "{1,1,2}}}",
           "~6: {{{-0.866,-0.866,1.13}, {0.866,0.866,2.87}}, {{-1,-1,1}, "
           "{1,1,3}}}",
           "7: {{{-0.866,-0.866,1.13}, {0.866,0.866,2.87}}, {{-1,-1,1}, "
           "{1,1,3}}}",
           "8: {null, inf}"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
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
    static char const* const expected_bound_strings[]
        = {"3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "~4: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, "
           "{{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}}",
           "5: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, "
           "{{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}}",
           "6: {null, inf}"};
    static char const* const expected_trans_strings[]
        = {"3: t=0 -> {}", "4: t=0", "5: t=1 -> {{0,0,1}}", "6: t=0"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
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
    static char const* const expected_bound_strings[]
        = {"~2: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "3: {{{-0.866,-0.866,-0.866}, {0.866,0.866,0.866}}, {{-1,-1,-1}, "
           "{1,1,1}}}",
           "~4: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, "
           "{{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}}",
           "5: {{{-0.433,-0.433,0.567}, {0.433,0.433,1.43}}, "
           "{{-0.5,-0.5,0.5}, {0.5,0.5,1.5}}}",
           "6: {null, inf}",
           "~7: {{{-1.08,-1.08,2.92}, {1.08,1.08,5.08}}, {{-1.25,-1.25,2.75}, "
           "{1.25,1.25,5.25}}}",
           "8: {{{-1.08,-1.08,2.92}, {1.08,1.08,5.08}}, {{-1.25,-1.25,2.75}, "
           "{1.25,1.25,5.25}}}",
           "9: {null, inf}",
           "10: {{{-0.433,-0.433,0.567}, {0.433,0.433,0.866}}, "
           "{{-0.5,-0.5,0.5}, {0.5,0.5,1}}}"};
    static char const* const expected_trans_strings[] = {"2: t=0 -> {}",
                                                         "3: t=0",
                                                         "4: t=0",
                                                         "5: t=1 -> {{0,0,1}}",
                                                         "6: t=0",
                                                         "7: t=0",
                                                         "8: t=2 -> {{0,0,4}}",
                                                         "9: t=0",
                                                         "10: t=0"};
    static int const expected_volume_nodes[] = {6, 8, 9, 10};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
