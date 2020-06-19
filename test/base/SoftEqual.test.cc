//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SoftEqual.test.cc
//---------------------------------------------------------------------------//
#include "base/SoftEqual.hh"

#include <limits>
#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::SoftEqual;
using celeritas::SoftZero;

//---------------------------------------------------------------------------//
TEST(SoftEqual, default_precisions)
{
    using Comp_t = SoftEqual<double, double>;

    EXPECT_DOUBLE_EQ(1e-12, Comp_t().rel());
    EXPECT_DOUBLE_EQ(1e-14, Comp_t().abs());

    EXPECT_DOUBLE_EQ(1e-6, Comp_t(1e-6).rel());
    EXPECT_DOUBLE_EQ(1e-8, Comp_t(1e-6).abs());

    EXPECT_DOUBLE_EQ(1e-4, Comp_t(1e-4, 1e-9).rel());
    EXPECT_DOUBLE_EQ(1e-9, Comp_t(1e-4, 1e-9).abs());
}

//---------------------------------------------------------------------------//
// Test fixture
//---------------------------------------------------------------------------//
template<typename T>
class FloatingTest : public celeritas::Test
{
  protected:
    // >>> TYPE ALIASES
    using value_type = T;
    using Limits_t   = std::numeric_limits<value_type>;
};

using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(FloatingTest, FloatTypes);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TYPED_TEST(FloatingTest, soft_equal)
{
    using value_type = typename TestFixture::value_type;
    using Limits_t   = typename TestFixture::Limits_t;
    using Comp_t     = SoftEqual<value_type, value_type>;

    Comp_t comp;

    // Test basic equality
    EXPECT_TRUE(comp(1, 1));
    EXPECT_TRUE(comp(0, 0));
    EXPECT_FALSE(comp(-1, 1));
    EXPECT_FALSE(comp(1, -1));

    // Test with tolerance
    EXPECT_TRUE(comp(1, 1 + comp.rel() / 2));
    EXPECT_FALSE(comp(1, 1 + comp.rel() * 2));

    // TODO: absolute tolerace is wacky
    EXPECT_TRUE(comp(0, comp.rel() / 2));
    EXPECT_TRUE(comp(comp.abs() / 2, comp.rel() / 2));
    EXPECT_FALSE(comp(0, comp.rel()));
    EXPECT_FALSE(comp(comp.abs(), comp.rel() / 2));
    EXPECT_FALSE(comp(comp.abs(), comp.rel()));

    // Test signed zeros
    EXPECT_FALSE(comp(-0, 1));
    EXPECT_FALSE(comp(1, -0));
    EXPECT_TRUE(comp(0, -0));
    EXPECT_TRUE(comp(-0, 0));

    // Test NaNs
    const value_type nan = Limits_t::quiet_NaN();
    EXPECT_FALSE(comp(1, nan));
    EXPECT_FALSE(comp(nan, 1));
    EXPECT_FALSE(comp(nan, nan));

    // Test infinities
    const value_type inf    = Limits_t::infinity();
    const value_type maxval = Limits_t::max();
    EXPECT_FALSE(comp(0, inf));
    EXPECT_FALSE(comp(inf, 0));
    EXPECT_TRUE(comp(inf, inf));
    EXPECT_FALSE(comp(inf, -inf));
    EXPECT_FALSE(comp(-inf, inf));
    EXPECT_FALSE(comp(inf, maxval));
}

//---------------------------------------------------------------------------//
TYPED_TEST(FloatingTest, soft_zero)
{
    using value_type = typename TestFixture::value_type;
    using Limits_t   = typename TestFixture::Limits_t;
    using Comp_t     = SoftZero<value_type>;

    Comp_t comp;

    // Test basic equality
    EXPECT_TRUE(comp(0));
    EXPECT_FALSE(comp(-1));
    EXPECT_FALSE(comp(1));

    // Test with tolerance
    EXPECT_FALSE(comp(comp.abs()));
    EXPECT_FALSE(comp(-comp.abs()));
    EXPECT_TRUE(comp(comp.abs() / 2));
    EXPECT_TRUE(comp(-comp.abs() / 2));

    // Test signed zeros
    EXPECT_TRUE(comp(-0));

    // Test NaNs
    const value_type nan = Limits_t::quiet_NaN();
    EXPECT_FALSE(comp(nan));

    // Test infinities
    const value_type inf = Limits_t::infinity();
    EXPECT_FALSE(comp(inf));
    EXPECT_FALSE(comp(-inf));
}

//---------------------------------------------------------------------------//
// Test fixture
//---------------------------------------------------------------------------//
template<typename PairT>
class MixedTest : public celeritas::Test
{
  protected:
    // >>> TYPE ALIASES
    using Comp_t
        = SoftEqual<typename PairT::first_type, typename PairT::second_type>;
    using value_type = typename Comp_t::value_type;
    using Limits_t   = std::numeric_limits<value_type>;
};

using MixedTypes
    = ::testing::Types<std::pair<float, double>, std::pair<double, float>>;
TYPED_TEST_SUITE(MixedTest, MixedTypes);

//---------------------------------------------------------------------------//
TYPED_TEST(MixedTest, comparison)
{
    using value_type = typename TestFixture::value_type;
    using Comp_t     = typename TestFixture::Comp_t;

    Comp_t comp;

    // Check types
    EXPECT_STREQ(typeid(float).name(), typeid(value_type).name());
    EXPECT_FLOAT_EQ(1.e-6, comp.rel());
    EXPECT_FLOAT_EQ(1.e-8, comp.abs());
}
