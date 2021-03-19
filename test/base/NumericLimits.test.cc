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
#include "celeritas_test.hh"
#include "NumericLimits.test.hh"

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// REAL TYPES
//---------------------------------------------------------------------------//

template<class T>
class RealNumericLimitsTest : public celeritas::Test
{
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RealNumericLimitsTest, RealTypes);

TYPED_TEST(RealNumericLimitsTest, host)
{
    using std_limits_t   = std::numeric_limits<TypeParam>;
    using celer_limits_t = celeritas::numeric_limits<TypeParam>;

    EXPECT_EQ(std_limits_t::epsilon(), celer_limits_t::epsilon());
    EXPECT_EQ(std_limits_t::infinity(), celer_limits_t::infinity());
    EXPECT_EQ(std_limits_t::max(), celer_limits_t::max());
    EXPECT_TRUE(std::isnan(celer_limits_t::quiet_NaN()));
}

TYPED_TEST(RealNumericLimitsTest, device)
{
    using celer_limits_t = celeritas::numeric_limits<TypeParam>;
    auto result          = nl_test<TypeParam>();

    EXPECT_EQ(celer_limits_t::epsilon(), result.eps);
    EXPECT_TRUE(std::isnan(result.nan));
    EXPECT_EQ(celer_limits_t::infinity(), result.inf);
    EXPECT_EQ(celer_limits_t::max(), result.max);
}

//---------------------------------------------------------------------------//
// UNSIGNED INT TYPES
//---------------------------------------------------------------------------//

template<class T>
class UIntNumericLimitsTest : public celeritas::Test
{
};

using UIntTypes
    = ::testing::Types<unsigned int, unsigned long, unsigned long long>;
TYPED_TEST_SUITE(UIntNumericLimitsTest, UIntTypes);

TYPED_TEST(UIntNumericLimitsTest, host)
{
    using std_limits_t   = std::numeric_limits<TypeParam>;
    using celer_limits_t = celeritas::numeric_limits<TypeParam>;

    EXPECT_EQ(std_limits_t::max(), celer_limits_t::max());
}
