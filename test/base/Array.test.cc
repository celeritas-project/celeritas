//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.test.cc
//---------------------------------------------------------------------------//
#include "base/Array.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::array;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(ArrayTest, all)
{
    // Note: C++14 required to write without double brackets
    array<int, 3> x = {1, 3, 2};

    EXPECT_FALSE(x.empty());
    EXPECT_EQ(3, x.size());
    EXPECT_EQ(1, x.front());
    EXPECT_EQ(2, x.back());

    array<int, 3> y{20, 30, 40};
    EXPECT_EQ(x, x);
    EXPECT_NE(x, y);
}
