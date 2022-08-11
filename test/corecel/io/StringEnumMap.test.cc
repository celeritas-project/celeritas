//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringEnumMap.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/StringEnumMap.hh"

#include <algorithm>
#include <cctype>

#include "corecel/Assert.hh"

#include "celeritas_test.hh"

using celeritas::StringEnumMap;

//---------------------------------------------------------------------------//
enum class SomeOtherEnum;

enum class CeleritasLabs
{
    argonne,
    fermilab,
    ornl,
    size_
};

const char* to_cstring(CeleritasLabs lab)
{
    CELER_EXPECT(lab != CeleritasLabs::size_);
    static const char* strings[] = {"argonne", "fermilab", "ornl"};
    return strings[static_cast<int>(lab)];
}

// Make sure to test in the case where to_cstring is overloaded
const char* to_cstring(SomeOtherEnum);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(StringEnumMapTest, from_cstring)
{
    auto from_string = StringEnumMap<CeleritasLabs>::from_cstring_func(
        to_cstring, "lab name");

    EXPECT_EQ(CeleritasLabs::argonne, from_string("argonne"));
    EXPECT_EQ(CeleritasLabs::fermilab, from_string("fermilab"));
    EXPECT_EQ(CeleritasLabs::ornl, from_string("ornl"));
    EXPECT_THROW(from_string("inl"), celeritas::RuntimeError);
}

TEST(StringEnumMapTest, from_generic)
{
    auto capstring = [](CeleritasLabs lab) -> std::string {
        std::string temp = to_cstring(lab);
        std::transform(
            temp.begin(), temp.end(), temp.begin(), [](unsigned char c) {
                return std::toupper(c);
            });
        return temp;
    };

    StringEnumMap<CeleritasLabs> from_string(capstring);

    EXPECT_EQ(CeleritasLabs::argonne, from_string("ARGONNE"));
    EXPECT_EQ(CeleritasLabs::fermilab, from_string("FERMILAB"));
    EXPECT_EQ(CeleritasLabs::ornl, from_string("ORNL"));
    EXPECT_THROW(from_string("ornl"), celeritas::RuntimeError);
}
