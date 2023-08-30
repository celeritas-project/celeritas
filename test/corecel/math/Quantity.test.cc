//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Quantity.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Quantity.hh"

#include <type_traits>

#include "celeritas_config.h"
#include "corecel/math/Turn.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_JSON
#    include "corecel/math/QuantityIO.json.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

constexpr double pi = m_pi;

// One revolution = 2pi radians
struct TwoPi
{
    static double value() { return 2 * pi; }
};
using Revolution = Quantity<TwoPi, double>;

struct DozenUnit
{
    static constexpr int value() { return 12; }
    static constexpr char const* label() { return "dozen"; }
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
    double dtheta = user_input.value() / 8;
    EXPECT_EQ(1.0 / 16.0, dtheta);

    // Hypothetical return value for user
    Revolution spacing{dtheta};
    EXPECT_SOFT_EQ(2 * pi / 16, native_value_from(spacing));

    // Create a quantity from a literal value in the native unit system
    auto half_rev = native_value_to<Revolution>(pi);
    EXPECT_TRUE((std::is_same<decltype(half_rev), Revolution>::value));
    EXPECT_DOUBLE_EQ(0.5, value_as<Revolution>(half_rev));

    // Check integer division works correctly
    using Dozen = Quantity<DozenUnit, int>;
    auto two_dozen = native_value_to<Dozen>(24);
    EXPECT_TRUE((std::is_same_v<decltype(two_dozen), Dozen>));
    EXPECT_EQ(2, value_as<Dozen>(two_dozen));

    auto twentyfour = native_value_from(two_dozen);
    EXPECT_TRUE((std::is_same_v<decltype(twentyfour), int>));
    EXPECT_EQ(24, twentyfour);
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

TEST(QuantityTest, mixed_precision)
{
    using RevInt = Quantity<TwoPi, int>;
    auto fourpi = native_value_from(RevInt{2});
    EXPECT_TRUE((std::is_same_v<decltype(fourpi), double>));
    EXPECT_SOFT_EQ(4 * pi, fourpi);

    using DozenDbl = Quantity<DozenUnit, double>;
    auto two_dozen = native_value_to<DozenDbl>(24);
    EXPECT_TRUE((std::is_same_v<decltype(two_dozen), DozenDbl>));
    EXPECT_SOFT_EQ(2, two_dozen.value());
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

    EXPECT_TRUE((Quantity<DozenUnit, int>{5} == Quantity<DozenUnit, long>{5}));
}

TEST(QuantityTest, unitless)
{
    EXPECT_TRUE(neg_max_quantity() < Revolution{-1e300});
    EXPECT_TRUE(neg_max_quantity() < zero_quantity());
    EXPECT_TRUE(zero_quantity() < max_quantity());
    EXPECT_TRUE(max_quantity() > Revolution{1e300});
}

TEST(QuantityTest, math)
{
    using RevInt = Quantity<TwoPi, int>;
    using RevFlt = Quantity<TwoPi, float>;
    using RevDbl = Quantity<TwoPi, double>;

    {
        auto added = RevDbl{1.5} + RevDbl{2.5};
        EXPECT_TRUE((std::is_same<decltype(added), RevDbl>::value));
        EXPECT_DOUBLE_EQ(4, added.value());
    }

    {
        auto subbed = RevFlt{1.5} - RevFlt{2.5};
        EXPECT_TRUE((std::is_same<decltype(subbed), RevFlt>::value));
        EXPECT_FLOAT_EQ(-1.0, subbed.value());
    }

    {
        auto negated = -RevDbl{1.5};
        EXPECT_TRUE((std::is_same<decltype(negated), RevDbl>::value));
        EXPECT_DOUBLE_EQ(-1.5, negated.value());
    }

    {
        auto muld = RevDbl{3} * 4;
        EXPECT_TRUE((std::is_same<decltype(muld), RevDbl>::value));
        EXPECT_DOUBLE_EQ(12, muld.value());
    }

    {
        auto divd = RevDbl{12} / 4;
        EXPECT_TRUE((std::is_same<decltype(divd), RevDbl>::value));
        EXPECT_DOUBLE_EQ(3, divd.value());
    }

    // Test mixed precision
    {
        EXPECT_DOUBLE_EQ(4 * pi, native_value_from(RevInt{2}));
        auto added = RevFlt{1.5} + RevInt{1};
        EXPECT_TRUE((std::is_same<decltype(added), RevFlt>::value));
    }
    {
        auto muld = RevInt{3} * 1.5;
        EXPECT_TRUE((std::is_same<decltype(muld), RevDbl>::value));
        EXPECT_DOUBLE_EQ(4.5, muld.value());
    }
}

TEST(QuantityTest, swappiness)
{
    using Dozen = Quantity<DozenUnit, int>;
    Dozen dozen{1}, gross{12};
    {
        // ADL should prefer swap implementation
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

TEST(QuantityTest, TEST_IF_CELERITAS_JSON(io))
{
#if CELERITAS_USE_JSON
    using Dozen = Quantity<DozenUnit, int>;

    {
        SCOPED_TRACE("Input as scalar");
        nlohmann::json inp = int{123};
        auto result = inp.get<Dozen>();
        EXPECT_EQ(123, value_as<Dozen>(result));
    }
    {
        SCOPED_TRACE("Input as array");
        nlohmann::json inp = {123, "dozen"};
        auto result = inp.get<Dozen>();
        EXPECT_EQ(123, value_as<Dozen>(result));
    }
    {
        SCOPED_TRACE("Invalid array size");
        nlohmann::json inp{{123, 456, 789}};
        EXPECT_THROW(inp.get<Dozen>(), RuntimeError);
    }
    {
        SCOPED_TRACE("Invalid unit");
        nlohmann::json inp = {123, "baker's dozen"};
        EXPECT_THROW(inp.get<Dozen>(), RuntimeError);
    }
    {
        SCOPED_TRACE("Output");
        nlohmann::json out = Dozen{2};
        static char const expected[] = R"json([2,"dozen"])json";
        EXPECT_EQ(std::string(expected), std::string(out.dump()));
    }
#endif
}

TEST(TurnTest, basic)
{
    EXPECT_EQ("tr", Turn::unit_type::label());
    EXPECT_SOFT_EQ(0.5, Turn{0.5}.value());
    EXPECT_SOFT_EQ(2 * pi, native_value_from(Turn{1}));
}

TEST(TurnTest, math)
{
    EXPECT_EQ(real_type(1), sin(Turn{0.25}));
    EXPECT_EQ(real_type(-1), cos(Turn{0.5}));
    EXPECT_EQ(real_type(0), sin(Turn{0}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
