//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/EnumStringMapper.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/EnumStringMapper.hh"

#include <sstream>

#include "celeritas_test.hh"
// #include "EnumStringMapper.test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
enum class Labs
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

char const* to_cstring(Labs value)
{
    static EnumStringMapper<Labs> const to_cstring_impl{
        "argonne",
        "fermilab",
        "ornl",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//

TEST(EnumStringMapperTest, all)
{
    EXPECT_STREQ("argonne", to_cstring(Labs::argonne));
    EXPECT_STREQ("fermilab", to_cstring(Labs::fermilab));
    EXPECT_STREQ("ornl", to_cstring(Labs::ornl));
    EXPECT_TRUE(std::string{to_cstring(Labs::size_)}.find("invalid")
                != std::string::npos);
}

TEST(EnumStringMapperTest, ostream)
{
    std::ostringstream msg;
    msg << Labs::argonne << ", " << Labs::fermilab << " and " << Labs::ornl;
    EXPECT_EQ("argonne, fermilab and ornl", msg.str());
}

// The following instances should fail to compile.
#if 0
TEST(EnumStringMapperTest, compiler_error)
{
    static EnumStringMapper<Labs> const too_short{"argonne", "ornl"};
    static EnumStringMapper<Labs> const too_long{
        "argonne", "ornl", "foo", "bar"};
    static EnumStringMapper<InvalidEnum> const no_size{"foo", "bar"};
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
