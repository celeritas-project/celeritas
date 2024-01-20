//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Version.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/Version.hh"

#include <string_view>

#include "celeritas_test.hh"

#define VT_G4VERSION 1063
static char const vt_g4version[] = "10.6.3";
#define VT_CELERITAS_VERSION 0x010203

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(VersionTest, constructors)
{
    {
        constexpr Version v{1, 2, 3};
        EXPECT_EQ(1, v.major());
        EXPECT_EQ(2, v.minor());
        EXPECT_EQ(3, v.patch());
    }
    {
        constexpr Version v{1, 2};
        EXPECT_EQ(1, v.major());
        EXPECT_EQ(2, v.minor());
        EXPECT_EQ(0, v.patch());
    }
    {
        constexpr Version v{1};
        EXPECT_EQ(1, v.major());
        EXPECT_EQ(0, v.minor());
        EXPECT_EQ(0, v.patch());
    }
    {
        constexpr auto v = Version::from_hex_xxyyzz(VT_CELERITAS_VERSION);
        EXPECT_EQ(1, v.major());
        EXPECT_EQ(2, v.minor());
        EXPECT_EQ(3, v.patch());
    }
    {
        constexpr auto v = Version::from_dec_xyz(VT_G4VERSION);
        EXPECT_EQ(10, v.major());
        EXPECT_EQ(6, v.minor());
        EXPECT_EQ(3, v.patch());
    }
}

TEST(VersionTest, ordering)
{
    EXPECT_EQ(Version(1), Version(1, 0, 0));
    EXPECT_NE(Version(1, 0, 0), Version(1, 0, 1));
    EXPECT_LT(Version(1), Version(2));
    EXPECT_LT(Version(1, 0, 0), Version(1, 0, 1));
    EXPECT_LT(Version(1, 2, 3), Version(2, 0, 1));
    EXPECT_GT(Version(2, 0, 1), Version(1, 2, 3));
}

TEST(VersionTest, from_string)
{
    using namespace std::literals;
    EXPECT_EQ(Version(10, 6, 3), Version::from_string(vt_g4version));
    EXPECT_EQ(Version(1, 9), Version::from_string("1.9"sv));
    EXPECT_EQ(Version(1000, 99, 8), Version::from_string("1000.99.8.7"sv));
    EXPECT_EQ(Version(0), Version::from_string("0"));
    EXPECT_EQ(Version(0, 1), Version::from_string("0.1"sv));
    EXPECT_EQ(Version(0, 3, 1), Version::from_string("0.3.1-dev.2"sv));

    EXPECT_THROW(Version::from_string(""sv), celeritas::RuntimeError);
    EXPECT_THROW(Version::from_string("0.x"sv), celeritas::RuntimeError);
    EXPECT_THROW(Version::from_string("1.-2.3"sv), celeritas::RuntimeError);
    EXPECT_THROW(Version::from_string("nope"sv), celeritas::RuntimeError);
    EXPECT_THROW(Version::from_string("0.3.1blakjsdf"sv),
                 celeritas::RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
