//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/SoftEqual.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/SoftEqual.hh"

#include <limits>

#include "corecel/io/ColorUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(SoftEqual, default_precisions)
{
    using Comp = SoftEqual<double>;

    // Check that defaults can be accessed as constexpr
    constexpr auto rel = Comp().rel();
    constexpr auto abs = Comp().abs();

    EXPECT_DOUBLE_EQ(1e-12, rel);
    EXPECT_DOUBLE_EQ(1e-14, abs);

    EXPECT_DOUBLE_EQ(1e-6, Comp(1e-6).rel());
    EXPECT_DOUBLE_EQ(1e-8, Comp(1e-6).abs());

    EXPECT_DOUBLE_EQ(1e-4, Comp(1e-4, 1e-9).rel());
    EXPECT_DOUBLE_EQ(1e-9, Comp(1e-4, 1e-9).abs());
}

//---------------------------------------------------------------------------//
// Test fixture
//---------------------------------------------------------------------------//
template<class T>
class FloatingTest : public Test
{
  protected:
    using value_type = T;
    using Limits_t = std::numeric_limits<value_type>;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(FloatingTest, FloatTypes, );

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TYPED_TEST(FloatingTest, soft_equal)
{
    using value_type = typename TestFixture::value_type;
    using Limits_t = typename TestFixture::Limits_t;
    using Comp = SoftEqual<value_type>;

    Comp comp;

    // Test basic equality
    EXPECT_TRUE(comp(1, 1));
    EXPECT_TRUE(comp(0, 0));
    EXPECT_FALSE(comp(-1, 1));
    EXPECT_FALSE(comp(1, -1));

    // Test with tolerance
    EXPECT_TRUE(comp(1, 1 + comp.rel() / 2));
    EXPECT_FALSE(comp(1, 1 + comp.rel() * 2));
    // Large scale
    EXPECT_TRUE(comp(1e6, 1e6 * (1 + comp.rel() / 2)));
    EXPECT_FALSE(comp(1e6, 1e6 * (1 + comp.rel() * 2)));
    // Smaller scale
    EXPECT_TRUE(comp(1e-5, 1e-5 * (1 + comp.rel() / 2)));
    EXPECT_FALSE(comp(1e-5, 1e-6));

    if (std::is_same_v<double, value_type>)
    {
        // Double precision uses different relative/absolute tolerances
        EXPECT_FALSE(comp(0, comp.rel() / 2));
        EXPECT_FALSE(comp(comp.abs() / 2, comp.rel() / 2));
        EXPECT_FALSE(comp(0, comp.rel()));
        EXPECT_FALSE(comp(comp.abs(), comp.rel() / 2));
        EXPECT_FALSE(comp(comp.abs(), comp.rel()));
    }

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
    const value_type inf = Limits_t::infinity();
    const value_type maxval = Limits_t::max();
    EXPECT_FALSE(comp(0, inf));
    EXPECT_FALSE(comp(inf, 0));
    EXPECT_FALSE(comp(inf, -inf));
    EXPECT_FALSE(comp(-inf, inf));
    EXPECT_FALSE(comp(inf, maxval));

    // NOTE: values that are legitimately infinite require additional testing
    // outside of soft equal because they're an edge case.
    EXPECT_FALSE(comp(inf, inf));
}

TYPED_TEST(FloatingTest, equal_or_soft_equal)
{
    using value_type = typename TestFixture::value_type;
    using Limits_t = typename TestFixture::Limits_t;
    const value_type nan = Limits_t::quiet_NaN();
    const value_type inf = Limits_t::infinity();

    EqualOr<SoftEqual<value_type>> comp;

    EXPECT_TRUE(comp(1, 1));
    EXPECT_TRUE(comp(0, 0));
    EXPECT_FALSE(comp(-1, 1));
    EXPECT_FALSE(comp(1, -1));
    EXPECT_FALSE(comp(inf, -inf));
    EXPECT_FALSE(comp(-inf, inf));

    EXPECT_TRUE(comp(inf, inf));
    EXPECT_TRUE(comp(1e6, 1e6 * (1 + comp.rel() / 2)));
    EXPECT_FALSE(comp(1e6, 1e6 * (1 + comp.rel() * 2)));

    EXPECT_FALSE(comp(1, nan));
}

TYPED_TEST(FloatingTest, soft_zero)
{
    using value_type = typename TestFixture::value_type;
    using Limits_t = typename TestFixture::Limits_t;
    using Comp = SoftZero<value_type>;

    Comp comp;

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

TYPED_TEST(FloatingTest, soft_mod)
{
    using value_type = typename TestFixture::value_type;

    EXPECT_TRUE(soft_mod(value_type(1.0), value_type(0.25)));
    EXPECT_FALSE(soft_mod(value_type(1.0), value_type(0.8)));

    auto tol = SoftEqual<value_type>().abs() / 2;
    EXPECT_TRUE(soft_mod(value_type(1), value_type(0.25)));
    EXPECT_TRUE(soft_mod(value_type(3.6) + tol, value_type(1.2)));
    EXPECT_TRUE(soft_mod(value_type(3.6) - tol, value_type(1.2)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
