//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Label.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/Label.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(LabelTest, ordering)
{
    EXPECT_EQ(Label("a"), Label("a"));
    EXPECT_EQ(Label("a", "1"), Label("a", "1"));
    EXPECT_NE(Label("a"), Label("b"));
    EXPECT_NE(Label("a", "1"), Label("a", "2"));
    EXPECT_TRUE(Label("a") < Label("b"));
    EXPECT_FALSE(Label("a") < Label("a"));
    EXPECT_FALSE(Label("b") < Label("a"));
    EXPECT_TRUE(Label("a") < Label("a", "1"));
    EXPECT_TRUE(Label("a", "0") < Label("a", "1"));
    EXPECT_FALSE(Label("a", "1") < Label("a", "1"));
    EXPECT_FALSE(Label("a", "2") < Label("a", "1"));
}

TEST(LabelTest, construction)
{
    EXPECT_EQ(Label("foo"), Label::from_geant("foo"));
    EXPECT_EQ(Label("foo", "0xdeadb01d"), Label::from_geant("foo0xdeadb01d"));
    EXPECT_EQ(Label("foo", "0x1234"), Label::from_geant("foo0x1234"));
    EXPECT_EQ(Label("foo", "0x1e0cea00x1e0c5c0"),
              Label::from_geant("foo0x1e0cea00x1e0c5c0"));
    EXPECT_EQ(Label("foo", "0x1e0c8c0_refl"),
              Label::from_geant("foo0x1e0c8c0_refl"));

    EXPECT_EQ(Label("bar"), Label::from_separator("bar", '@'));
    EXPECT_EQ(Label("bar"), Label::from_separator("bar@", '@'));
    EXPECT_EQ(Label("bar", "123"), Label::from_separator("bar@123", '@'));
}

TEST(LabelTest, output)
{
    std::ostringstream os;
    os << Label{"bar", "123"};
    EXPECT_EQ("bar@123", os.str());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
