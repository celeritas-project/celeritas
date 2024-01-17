//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/EnumStringMapper.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/EnumStringMapper.hh"

#include "celeritas_test.hh"
// #include "EnumStringMapper.test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
enum class CeleritasLabs
{
    argonne,
    fermilab,
    ornl,
    size_
};

enum class InvalidEnum
{
    foo,
    bar
};

//---------------------------------------------------------------------------//

TEST(EnumStringMapperTest, all)
{
    static EnumStringMapper<CeleritasLabs> const to_string{
        "argonne", "fermilab", "ornl"};

    EXPECT_STREQ("argonne", to_string(CeleritasLabs::argonne));
    EXPECT_STREQ("fermilab", to_string(CeleritasLabs::fermilab));
    EXPECT_STREQ("ornl", to_string(CeleritasLabs::ornl));
    EXPECT_TRUE(std::string{to_string(CeleritasLabs::size_)}.find("invalid")
                != std::string::npos);
}

// The following instances should fail to compile.
#if 0
TEST(EnumStringMapperTest, compiler_error)
{
    static EnumStringMapper<CeleritasLabs> const too_short{"argonne", "ornl"};
    static EnumStringMapper<CeleritasLabs> const too_long{
        "argonne", "ornl", "foo", "bar"};
    static EnumStringMapper<InvalidEnum> const no_size{"foo", "bar"};
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
