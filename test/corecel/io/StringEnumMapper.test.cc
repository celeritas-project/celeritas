//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringEnumMapper.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/StringEnumMapper.hh"

#include <algorithm>
#include <cctype>

#include "corecel/Assert.hh"
#include "corecel/cont/EnumArray.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
enum class SomeOtherEnum;

enum class CeleritasLabs
{
    argonne,
    fermilab,
    ornl,
    size_
};

char const* to_cstring(CeleritasLabs lab)
{
    CELER_EXPECT(lab != CeleritasLabs::size_);
    static char const* strings[] = {"argonne", "fermilab", "ornl"};
    return strings[static_cast<int>(lab)];
}

// Make sure to test in the case where to_cstring is overloaded
char const* to_cstring(SomeOtherEnum);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(StringEnumMapperTest, from_cstring)
{
    auto from_string = StringEnumMapper<CeleritasLabs>::from_cstring_func(
        to_cstring, "lab name");

    EXPECT_EQ(CeleritasLabs::argonne, from_string("argonne"));
    EXPECT_EQ(CeleritasLabs::fermilab, from_string("fermilab"));
    EXPECT_EQ(CeleritasLabs::ornl, from_string("ornl"));
    EXPECT_THROW(from_string("inl"), RuntimeError);
}

TEST(StringEnumMapperTest, from_generic)
{
    // NOTE: string storage must exceed lifetime of string enum mapper
    EnumArray<CeleritasLabs, std::string> storage;
    auto capstring = [&storage](CeleritasLabs lab) -> std::string_view {
        std::string temp = to_cstring(lab);
        std::transform(
            temp.begin(), temp.end(), temp.begin(), [](unsigned char c) {
                return std::toupper(c);
            });
        storage[lab] = std::move(temp);
        return storage[lab];
    };

    StringEnumMapper<CeleritasLabs> from_string(capstring);

    EXPECT_EQ(CeleritasLabs::argonne, from_string("ARGONNE"));
    EXPECT_EQ(CeleritasLabs::fermilab, from_string("FERMILAB"));
    EXPECT_EQ(CeleritasLabs::ornl, from_string("ORNL"));
    EXPECT_THROW(from_string("ornl"), RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
