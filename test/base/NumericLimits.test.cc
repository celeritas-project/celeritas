//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NumericLimits.test.cc
//---------------------------------------------------------------------------//
#include "base/NumericLimits.hh"

#include <cmath>
#include <limits>
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "gmock/gmock.h"
#include "NumericLimits.test.hh"

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

template<class T>
class NumericLimitsTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NumericLimitsTest, RealTypes);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TYPED_TEST(NumericLimitsTest, all)
{
    using limits_t = celeritas::numeric_limits<TypeParam>;
    auto result    = nl_test<TypeParam>();

    using testing::ElementsAreArray;
    EXPECT_EQ(limits_t::epsilon(), result.eps);
    EXPECT_TRUE(std::isnan(result.nan));
    EXPECT_EQ(limits_t::infinity(), result.inf);
}
