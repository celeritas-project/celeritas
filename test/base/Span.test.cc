//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Span.test.cc
//---------------------------------------------------------------------------//
#include "base/Span.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::span;

using SpanDbl = celeritas::span<double>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SpanTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SpanTest, basic)
{
    // Test a default span<double>
    SpanDbl empty_span;
    EXPECT_EQ(empty_span.begin(), nullptr);
    EXPECT_EQ(empty_span.end(), nullptr);
    EXPECT_EQ(empty_span.data(), nullptr);

    // >>> Test a Span_Dbl with pointers

    // Make a field
    std::vector<double> y(4);
    fill(y.begin(), y.end(), 0.57);
    y[2] = 0.62;

    // Get a view
    {
        SpanDbl v(&y[0], &y[3] + 1);
        EXPECT_EQ(4, v.size());
        EXPECT_TRUE(!v.empty());

        EXPECT_EQ(0.57, v[0]);
        EXPECT_EQ(0.57, v[1]);
        EXPECT_EQ(0.62, v[2]);
        EXPECT_EQ(0.57, v[3]);

        // change a value
        v[1] = 0.23;
    }

    // Check field
    EXPECT_EQ(0.57, y[0]);
    EXPECT_EQ(0.23, y[1]);
    EXPECT_EQ(0.62, y[2]);
    EXPECT_EQ(0.57, y[3]);
    EXPECT_EQ(0.57, y.front());
    EXPECT_EQ(0.57, y.back());
}
