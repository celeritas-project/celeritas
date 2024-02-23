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
    Tol tolerance() const override { return Tol::from_relative(1e-4); }
};

using NegationTest = CsgObjectTest;

TEST_F(NegationTest, just_neg)
{
    auto sph = std::make_shared<SphereShape>("sph", 1.0);
    auto trsph = std::make_shared<Transformed>(sph, Translation{{0, 0, 1}});

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

TEST_F(NegationTest, pos_neg)
{
    auto sph = std::make_shared<SphereShape>("sph", 1.0);
    auto trsph = std::make_shared<Transformed>(sph, Translation{{0, 0, 1}});

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

TEST_F(NegationTest, double_neg)
{
    auto sph = std::make_shared<SphereShape>("sph", 1.0);
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
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
