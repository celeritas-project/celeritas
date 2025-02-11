//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
