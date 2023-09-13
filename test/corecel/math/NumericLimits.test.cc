//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/NumericLimits.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/NumericLimits.hh"

#include <cmath>
#include <limits>

#include "NumericLimits.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// REAL TYPES
//---------------------------------------------------------------------------//

template<class T>
class RealNumericLimitsTest : public Test
{
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RealNumericLimitsTest, RealTypes, );

TYPED_TEST(RealNumericLimitsTest, host)
{
    using std_limits_t = std::numeric_limits<TypeParam>;
    using celer_limits_t = numeric_limits<TypeParam>;

    EXPECT_EQ(std_limits_t::epsilon(), celer_limits_t::epsilon());
    EXPECT_EQ(std_limits_t::infinity(), celer_limits_t::infinity());
    EXPECT_EQ(std_limits_t::max(), celer_limits_t::max());
    EXPECT_TRUE(std::isnan(celer_limits_t::quiet_NaN()));
#ifndef _MSC_VER
    EXPECT_EQ(std_limits_t::infinity(), TypeParam(1) / TypeParam(0));
#endif
}

#if CELER_USE_DEVICE
TYPED_TEST(RealNumericLimitsTest, device)
#else
TYPED_TEST(RealNumericLimitsTest, DISABLED_device)
#endif
{
    using celer_limits_t = numeric_limits<TypeParam>;
    auto result = nl_test<TypeParam>();

    EXPECT_EQ(celer_limits_t::epsilon(), result.eps);
    EXPECT_TRUE(std::isnan(result.nan));
    EXPECT_EQ(celer_limits_t::infinity(), result.inf);
    EXPECT_EQ(celer_limits_t::max(), result.max);
    EXPECT_EQ(celer_limits_t::infinity(), result.inv_zero);
}

//---------------------------------------------------------------------------//
// UNSIGNED INT TYPES
//---------------------------------------------------------------------------//

template<class T>
class UIntNumericLimitsTest : public Test
{
};

using UIntTypes
    = ::testing::Types<unsigned int, unsigned long, unsigned long long>;
TYPED_TEST_SUITE(UIntNumericLimitsTest, UIntTypes, );

TYPED_TEST(UIntNumericLimitsTest, host)
{
    using std_limits_t = std::numeric_limits<TypeParam>;
    using celer_limits_t = numeric_limits<TypeParam>;

    EXPECT_EQ(std_limits_t::max(), celer_limits_t::max());
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
