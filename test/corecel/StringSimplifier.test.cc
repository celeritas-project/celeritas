//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/StringSimplifier.test.cc
//---------------------------------------------------------------------------//
#include "StringSimplifier.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(StringSimplifierTest, all)
{
    StringSimplifier simplify(4);
    EXPECT_EQ("This is a pointer 0x0 yep",
              simplify("This is a pointer 0x12345 yep"));

    EXPECT_EQ("And this value 0.1235 gets rounded",
              simplify("And this value 0.12345 gets rounded"));
    EXPECT_EQ("As does 3.4059e3 and even 1.01e1",
              simplify("As does 3.4059123e3 and even 1.01e1"));
    EXPECT_EQ("Single precision is 2., 1.23, or 1.678e-3 now",
              simplify("Single precision is 2.f, 1.23f, or 1.678e-3f now"));
    EXPECT_EQ("And finally we remove colors",
              simplify("And finally we remove \033[31;1mcolors\033[0m"));

    EXPECT_EQ("123.25", simplify("123.25"));
    EXPECT_EQ("123.254", simplify("123.254"));
    EXPECT_EQ("123.2541", simplify("123.2541"));
    EXPECT_EQ("123.2541", simplify("123.25412"));
    EXPECT_EQ("1.25e-2", simplify("1.25e-002"));
    EXPECT_EQ("1.25e-2", simplify("1.25e-02"));
    EXPECT_EQ("1.25e-2", simplify("1.25e-2"));
    EXPECT_EQ("1.25e-20", simplify("1.25e-20"));
    EXPECT_EQ("1.25e2", simplify("1.25e+2"));
    EXPECT_EQ("1.25e2", simplify("1.25e+02"));
    EXPECT_EQ("1.25e2", simplify("1.25e+002"));
    EXPECT_EQ("1.25e2", simplify("1.25e2"));
    EXPECT_EQ("1.254e2", simplify("1.254e2"));
    EXPECT_EQ("1.2541e2", simplify("1.2541e2"));
    EXPECT_EQ("1.2541e2", simplify("1.25412e2"));

    StringSimplifier simplify_more(1);
    EXPECT_EQ("And this value 0.1 gets rounded",
              simplify_more("And this value 0.12345 gets rounded"));
    EXPECT_EQ("12.3 1.0e0", simplify_more("12.3456 1.0234e0"));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
