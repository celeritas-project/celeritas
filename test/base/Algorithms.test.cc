//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.test.cc
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"

// using celeritas;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(AlgorithmsTest, all)
{
    // Test min()
    EXPECT_EQ(1, celeritas::min<int>(1, 2));
    EXPECT_NE(0.2, celeritas::min<float>(0.1, 0.2));
    // Test max()
    EXPECT_EQ(2, celeritas::max<int>(1, 2));
    EXPECT_NE(0.1, celeritas::max<float>(0.1, 0.2));
    // Test cube()
    EXPECT_EQ(8, celeritas::cube<int>(2));
    EXPECT_EQ(0.001f, celeritas::cube<float>(0.1));
    EXPECT_NE(125.000001, celeritas::cube<double>(5.0));
}
