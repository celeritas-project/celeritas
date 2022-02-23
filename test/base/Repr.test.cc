//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Repr.test.cc
//---------------------------------------------------------------------------//
#include "base/Repr.hh"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "base/Array.hh"
#include "base/Span.hh"

#include "celeritas_test.hh"

using celeritas::repr;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<class T, class... Args>
std::string repr_to_string(const T& obj, Args... args)
{
    std::ostringstream os;
    os << repr(obj, args...);
    return os.str();
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(ReprTest, fundamental)
{
    EXPECT_EQ("1.25f", repr_to_string(1.25f));
    // EXPECT_EQ("1.25f", repr_to_string(1.25f, "")); // TODO
    EXPECT_EQ("float foo{1.25f}", repr_to_string(1.25f, "foo"));

    EXPECT_EQ("1.25", repr_to_string(1.25));
    EXPECT_EQ("1", repr_to_string(1));
    EXPECT_EQ("1ull", repr_to_string(1ull));
    EXPECT_EQ("1ll", repr_to_string(1ll));
    EXPECT_EQ("'a'", repr_to_string('a'));
    EXPECT_EQ("'\\x61'", repr_to_string(static_cast<unsigned char>('a')));
    EXPECT_EQ("'\\xff'", repr_to_string('\xff'));
}

TEST(ReprTest, string)
{
    EXPECT_EQ("\"abcd\\a\"", repr_to_string("abcd\a"));
    EXPECT_EQ("\"abcd\\a\"", repr_to_string(std::string("abcd\a")));
    EXPECT_EQ("std::string hi{\"hello\"}",
              repr_to_string(std::string("hello"), "hi"));
}

TEST(ReprTest, container)
{
    EXPECT_EQ("{1, 2, 3, 4}", repr_to_string(std::vector<int>{1, 2, 3, 4}));
    EXPECT_EQ("{100l, 200l}",
              repr_to_string(celeritas::Array<long, 2>{100, 200}));

    unsigned int uints[] = {11, 22};
    EXPECT_EQ("{11u, 22u}", repr_to_string(celeritas::make_span(uints)));

    const char* const cstrings[] = {"one", "three", "five"};
    EXPECT_EQ("{\"one\", \"three\", \"five\"}", repr_to_string(cstrings));

    const std::string strings[] = {"a", "", "special\nchars\t"};
    EXPECT_EQ("{\"a\", \"\", \"special\\nchars\\t\"}", repr_to_string(strings));
}
