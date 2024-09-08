//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Color.test.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/Color.hh"

#include "celeritas_test.hh"

using size_type = celeritas::Color::size_type;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ColorTest : public ::celeritas::test::Test
{
  protected:
    using Channel = celeritas::Color::Channel;
};

TEST_F(ColorTest, default_constructor)
{
    Color c;
    EXPECT_EQ(0u, static_cast<Color::size_type>(c));
}

TEST_F(ColorTest, from_html)
{
    EXPECT_EQ(0x0178efffu,
              static_cast<Color::size_type>(Color::from_html("#0178ef")));
    EXPECT_EQ(0x0178ef12u,
              static_cast<Color::size_type>(Color::from_html("#0178ef12")));

    EXPECT_THROW(Color::from_html("#01z8ef"), RuntimeError);
    EXPECT_THROW(Color::from_html("0178ef"), RuntimeError);
}

TEST_F(ColorTest, from_rgb)
{
    Color c = Color::from_rgb(0x0178ef);
    EXPECT_EQ(0x0178efffu, static_cast<Color::size_type>(c));

    c = Color::from_rgb(0xffffff);
    EXPECT_EQ(0xffffffffu, static_cast<Color::size_type>(c));
}

TEST_F(ColorTest, from_rgba)
{
    Color c = Color::from_rgba(0x0178efff);
    EXPECT_EQ(0x0178efffu, static_cast<Color::size_type>(c));

    c = Color::from_rgba(0xffffff00);
    EXPECT_EQ(0xffffff00u, static_cast<Color::size_type>(c));
}

TEST_F(ColorTest, to_html)
{
    Color c = Color::from_rgb(0x0178ef);
    EXPECT_EQ("#0178ef", c.to_html());

    c = Color::from_rgba(0xffffffff);
    EXPECT_EQ("#ffffff", c.to_html());

    c = Color::from_rgba(0xffffff00);
    EXPECT_EQ("#ffffff00", c.to_html());

    c = Color::from_rgba(0x01ef7812);
    EXPECT_EQ("#01ef7812", c.to_html());
}

TEST_F(ColorTest, channel)
{
    Color c = Color::from_rgba(0x12345678);
    EXPECT_EQ(0x78, c.channel(Channel::alpha));
    EXPECT_EQ(0x56, c.channel(Channel::blue));
    EXPECT_EQ(0x34, c.channel(Channel::green));
    EXPECT_EQ(0x12, c.channel(Channel::red));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(c.channel(Channel::size_), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
