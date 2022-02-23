//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Quantity.test.cc
//---------------------------------------------------------------------------//
#include "base/Quantity.hh"

#include <type_traits>

#include "base/Constants.hh"

#include "celeritas_test.hh"

using celeritas::native_value_from;
using celeritas::native_value_to;
using celeritas::Quantity;
using celeritas::value_as;
using celeritas::zero_quantity;
using celeritas::constants::pi;

// One revolution = 2pi radians
struct TwoPi
{
    static double value() { return 2 * celeritas::constants::pi; }
};
using Revolution = Quantity<TwoPi, double>;

struct DozenUnit
{
    static int value() { return 12; }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(QuantityTest, simplicity)
{
    EXPECT_EQ(sizeof(Revolution), sizeof(double));
    EXPECT_TRUE(std::is_standard_layout<Revolution>::value);
    EXPECT_TRUE(std::is_default_constructible<Revolution>::value);
}

TEST(QuantityTest, usage)
{
    // Since powers of 2 are exactly represented in IEEE arithimetic, we can
    // exactly operate on data (e.g. in this case where a user wants a radial
    // mesh that spans half a turn, i.e. pi)
    Revolution user_input{0.5};
    double     dtheta = user_input.value() / 8;
    EXPECT_EQ(1.0 / 16.0, dtheta);

    // Hypothetical return value for user
    Revolution spacing{dtheta};
    EXPECT_SOFT_EQ(2 * pi / 16, native_value_from(spacing));

    // Create a quantity from a literal value in the native unit system
    auto half_rev = native_value_to<Revolution>(celeritas::constants::pi);
    EXPECT_TRUE((std::is_same<decltype(half_rev), Revolution>::value));
    EXPECT_DOUBLE_EQ(0.5, value_as<Revolution>(half_rev));
}

TEST(QuantityTest, zeros)
{
    // Construct a quantity with value of zero
    Revolution zero_turn;
    EXPECT_EQ(0, zero_turn.value());

    zero_turn = Revolution{10};

    // Construct from a "zero" sentinel type
    zero_turn = zero_quantity();
    EXPECT_EQ(0, value_as<Revolution>(zero_turn));
}

TEST(QuantityTest, comparators)
{
    EXPECT_TRUE(zero_quantity() < Revolution{4});
    EXPECT_TRUE(zero_quantity() <= Revolution{4});
    EXPECT_TRUE(zero_quantity() != Revolution{4});
    EXPECT_FALSE(zero_quantity() > Revolution{4});
    EXPECT_FALSE(zero_quantity() >= Revolution{4});
    EXPECT_FALSE(zero_quantity() == Revolution{4});

    EXPECT_TRUE(Revolution{3} < Revolution{4});
    EXPECT_TRUE(Revolution{3} <= Revolution{4});
    EXPECT_TRUE(Revolution{3} != Revolution{4});
    EXPECT_FALSE(Revolution{3} > Revolution{4});
    EXPECT_FALSE(Revolution{3} >= Revolution{4});
    EXPECT_FALSE(Revolution{3} == Revolution{4});

    EXPECT_FALSE(Revolution{5} < Revolution{4});
    EXPECT_FALSE(Revolution{5} <= Revolution{4});
    EXPECT_TRUE(Revolution{5} != Revolution{4});
    EXPECT_TRUE(Revolution{5} > Revolution{4});
    EXPECT_TRUE(Revolution{5} >= Revolution{4});
    EXPECT_FALSE(Revolution{5} == Revolution{4});
}

TEST(QuantityTest, infinities)
{
    using celeritas::max_quantity;
    using celeritas::neg_max_quantity;
    EXPECT_TRUE(neg_max_quantity() < Revolution{-1e300});
    EXPECT_TRUE(neg_max_quantity() < zero_quantity());
    EXPECT_TRUE(zero_quantity() < max_quantity());
    EXPECT_TRUE(max_quantity() > Revolution{1e300});
}

TEST(QuantityTest, swappiness)
{
    using Dozen = Quantity<DozenUnit, int>;
    Dozen dozen{1}, gross{12};
    {
        // ADL should prefer celeritas::swap implementation
        using std::swap;
        swap(dozen, gross);
        EXPECT_EQ(1, gross.value());
        EXPECT_EQ(12, dozen.value());
    }
    {
        // Should still work without std
        swap(dozen, gross);
        EXPECT_EQ(12, value_as<Dozen>(gross));
        EXPECT_EQ(1, value_as<Dozen>(dozen));
    }
    EXPECT_EQ(12, native_value_from(dozen));
    EXPECT_EQ(144, native_value_from(gross));
}
