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
    EXPECT_EQ("1e-3", simplify("0.001"));
    EXPECT_EQ("1e-3", simplify("0.001"));
    EXPECT_EQ("1e-3", simplify("0.001f"));
    EXPECT_EQ("-1e-3", simplify("-0.001"));
    EXPECT_EQ("-1e-3", simplify("-0.001f"));
    EXPECT_EQ("1.234", simplify("1.234"));
    EXPECT_EQ("12.34", simplify("12.34"));
    EXPECT_EQ("123.4", simplify("123.4"));
    EXPECT_EQ("1234.", simplify("1234."));
    EXPECT_EQ("1234.", simplify("1234.0"));
    EXPECT_EQ("1.234e4", simplify("12340."));
    EXPECT_EQ("0.1235", simplify("0.12345"));
    EXPECT_EQ("0.01235", simplify("0.012345"));

    EXPECT_EQ("This is a pointer 0x0 yep",
              simplify("This is a pointer 0x12345 yep"));

    EXPECT_EQ(R"(This "str123.0e5ing" better be ignored a f)",
              simplify(R"(This "str123.0e5ing" better be ignored a f)"));
    EXPECT_EQ("And this value 0.1235 gets rounded",
              simplify("And this value 0.12345 gets rounded"));
    EXPECT_EQ("As does 3.406e3 and even 1.01e1 and 1e7",
              simplify("As does 3.4059123e3 and even 1.01e1 and 1E+7"));
    EXPECT_EQ("Single precision is 1., 2., 1.23, or -1.678",
              simplify("Single precision is 1f, 2.f, 1.23f, or -1.678f"));
    EXPECT_EQ("Scientific single precision: 2e1, 3e2, 4.5e-1, -1e0",
              simplify("Scientific single precision: 2e1f, 3.e2f, 4.5e-1f, "
                       "-1e0f"));
    EXPECT_EQ("And finally we remove colors",
              simplify("And finally we remove \033[31;1mcolors\033[0m"));
    EXPECT_EQ("Zeros can be weird 0.0000 -0.0000 0.000 0e0 0.000e0 0e0 -0e0",
              simplify("Zeros can be weird 0.00000 -0.00000 00.000 0e0 "
                       "0.00000e0 0e1 -0e1"));
    EXPECT_EQ("Zero floats: 0.0000 -0.0000 0.000 0e0 0.000e0 0e0 -0e0",
              simplify("Zero floats: 0.00000f -0.00000f 00.000f 0e0f "
                       "0.00000e0f 0e1f -0e1f"));
    EXPECT_EQ("{-6., 0., 0.} along local direction {1., 0., 0.}",
              simplify("{-6f, 0f, 0f} along local direction {1f, 0f, 0f}"));
    EXPECT_EQ("{-6., 0., 0.} along local direction {1., 0., 0.}",
              simplify("{-6., 0., 0.} along local direction {1., 0., 0.}"));
    EXPECT_EQ("{-6, 0, 0} along local direction {1, 0, 0}",
              simplify("{-6, 0, 0} along local direction {1, 0, 0}"));
    EXPECT_EQ("Cone z: t=5e-3 at {0,0,100}",
              simplify("Cone z: t=0.005 at {0,0,100}"));

    EXPECT_EQ("0.0", simplify("0.0"));
    EXPECT_EQ("0.1", simplify("0.1"));
    EXPECT_EQ("0.01", simplify("0.01"));
    EXPECT_EQ("1e-3", simplify("0.001"));
    EXPECT_EQ("1e-4", simplify("0.0001"));
    EXPECT_EQ("1e-5", simplify("0.00001"));

    EXPECT_EQ("0.0", simplify("0.0f"));
    EXPECT_EQ("0.1", simplify("0.1f"));
    EXPECT_EQ("0.01", simplify("0.01f"));
    EXPECT_EQ("1e-3", simplify("0.001f"));
    EXPECT_EQ("1e-4", simplify("0.0001f"));
    EXPECT_EQ("1e-5", simplify("0.00001f"));

    EXPECT_EQ("123456", simplify("123456"));
    EXPECT_EQ("1234.", simplify("1234."));
    EXPECT_EQ("1234.", simplify("1234.f"));
    EXPECT_EQ("1.234e4", simplify("12345."));
    EXPECT_EQ("-123456", simplify("-123456"));
    EXPECT_EQ("-1234.", simplify("-1234."));
    EXPECT_EQ("-1234.", simplify("-1234.f"));
    EXPECT_EQ("-1.234e4", simplify("-12345."));
    EXPECT_EQ("123.3", simplify("123.251"));
    EXPECT_EQ("1.232", simplify("1.2319"));
    EXPECT_EQ("1.25e-2", simplify("1.25e-002"));
    EXPECT_EQ("1.25e-2", simplify("1.25e-02"));
    EXPECT_EQ("1.25e-2", simplify("1.25e-2"));
    EXPECT_EQ("1.25e-20", simplify("1.25e-20"));
    EXPECT_EQ("1.25e2", simplify("1.25e+2"));
    EXPECT_EQ("1.25e2", simplify("1.25e+02"));
    EXPECT_EQ("1.25e2", simplify("1.25e+002"));
    EXPECT_EQ("1.25e2", simplify("1.25e2"));
    EXPECT_EQ("1.254e2", simplify("1.254e2"));
    EXPECT_EQ("1.254e2", simplify("1.2541e2"));
    EXPECT_EQ("1.254e2", simplify("1.25412e2"));

    StringSimplifier simplify_more(2);
    EXPECT_EQ("And this value 0.12 gets rounded",
              simplify_more("And this value 0.12345 gets rounded"));
    EXPECT_EQ("12. 1.2e2 1.0e0", simplify_more("12.3456 123.45 1.0234e0"));

    StringSimplifier simplify_most(1);
    EXPECT_EQ("1e1 1e0", simplify_most("12.3456 1.0234e0"));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
