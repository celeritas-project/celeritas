//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Quantity.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Quantity.hh"

#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/math/QuantityIO.json.hh"
#include "corecel/math/Turn.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using constants::pi;

// One revolution = 2pi radians
struct TwoPi
{
    static constexpr Constant value() { return 2 * pi; }
};
using Revolution = Quantity<TwoPi, double>;

struct DozenUnit
{
    static constexpr int value() { return 12; }
    static constexpr char const* label() { return "dozen"; }
};

using Dozen = Quantity<DozenUnit, int>;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(QuantityTest, constexpr_attributes)
{
    EXPECT_TRUE((std::is_same_v<Revolution::value_type, double>));
    EXPECT_EQ(sizeof(Revolution), sizeof(double));
    EXPECT_TRUE(std::is_standard_layout<Revolution>::value);
    EXPECT_TRUE(std::is_default_constructible<Revolution>::value);

    EXPECT_TRUE((std::is_same_v<Dozen::value_type, int>));
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

    // Check int/untyped commparisons
    EXPECT_GT(Dozen{1}, zero_quantity());
    EXPECT_LT(Dozen{1}, max_quantity());
}

TEST(QuantityTest, mixed_precision)
{
    using RevInt = Quantity<TwoPi, int>;
    auto fourpi = native_value_from(RevInt{2});
    EXPECT_TRUE((std::is_same_v<decltype(fourpi), Constant>));
    EXPECT_SOFT_EQ(4 * pi, static_cast<double>(fourpi));

    using DozenDbl = Quantity<DozenUnit, double>;
    auto two_dozen = native_value_to<DozenDbl>(24);
    EXPECT_TRUE((std::is_same_v<decltype(two_dozen), DozenDbl>));
    EXPECT_SOFT_EQ(2, two_dozen.value());

    using DozenFlt = Quantity<DozenUnit, float>;
    {
        auto two_dozen_flt = native_value_to<DozenFlt>(24);
        EXPECT_SOFT_EQ(2, two_dozen_flt.value());
    }
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

    {
        auto divd = RevDbl{12} / RevDbl{3};
        EXPECT_TRUE((std::is_same<decltype(divd), double>::value));
        EXPECT_DOUBLE_EQ(4, divd);
    }

    // Test mixed integer/double
    {
        EXPECT_DOUBLE_EQ(static_cast<double>(4 * pi),
                         static_cast<double>(native_value_from(RevInt{2})));
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

TEST(QuantityTest, io)
{
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
}

TEST(TurnTest, basic)
{
    EXPECT_STREQ("tr", RealTurn::unit_type::label());
    EXPECT_SOFT_EQ(0.5, RealTurn{0.5}.value());
    EXPECT_REAL_EQ(static_cast<real_type>(2 * pi),
                   native_value_from(RealTurn{1}));
}

TEST(TurnTest, math)
{
    // Fractional powers of two should yield exact results
    EXPECT_EQ(double(1), sin(make_turn(0.25)));
    EXPECT_EQ(double(-1), cos(make_turn(0.5)));
    {
        auto result = sin(make_turn(0.0));
        EXPECT_TRUE((std::is_same_v<decltype(result), double>));
        EXPECT_EQ(double(0), result);
    }
    {
        auto turn = make_turn(2.0f);
        EXPECT_TRUE((std::is_same_v<decltype(turn), Turn_t<float>>));
        auto result = sin(turn);
        EXPECT_TRUE((std::is_same_v<decltype(result), float>));
        EXPECT_EQ(float(0), result);
    }
    {
        auto ta = atan2turn(0.0, 0.001);  // y, x
        EXPECT_TRUE((std::is_same<decltype(ta), Turn_t<double>>::value));
        EXPECT_DOUBLE_EQ(0.0, ta.value());
        ta = atan2turn(1.0, 0.0);
        EXPECT_DOUBLE_EQ(0.25, ta.value());
        ta = atan2turn(0.0, -1.0);
        EXPECT_DOUBLE_EQ(0.5, ta.value());
        ta = atan2turn(-0.0, -1.0);
        EXPECT_DOUBLE_EQ(-0.5, ta.value());
        ta = atan2turn(-1.0, 0.0);
        EXPECT_DOUBLE_EQ(-0.25, ta.value());
    }
    {
        auto t = make_turn(1.0 / 6);

        double s, c;
        sincos(t, &s, &c);
        EXPECT_DOUBLE_EQ(0.8660254037844386, s);
        EXPECT_DOUBLE_EQ(0.5, c);
        EXPECT_DOUBLE_EQ(1.7320508075688767, tan(t));
    }
}

TEST(QuarterTurnTest, basic)
{
    EXPECT_STREQ("qtr", IntQuarterTurn::unit_type::label());
    EXPECT_EQ(-1, IntQuarterTurn{-1}.value());
    EXPECT_EQ(1, IntQuarterTurn{1}.value());
    EXPECT_DOUBLE_EQ(static_cast<double>(2 * pi),
                     static_cast<double>(native_value_from(IntQuarterTurn{4})));
}

TEST(QuarterTurnTest, sincos)
{
    std::vector<int> actual;
    std::vector<int> expected;
    auto push_expected_int = [&expected](double v) {
        expected.push_back(static_cast<int>(std::round(v)));
    };

    for (auto i : range(-4, 5))
    {
        IntQuarterTurn theta{i};
        actual.push_back(sin(theta));
        actual.push_back(cos(theta));

        auto theta_dbl = static_cast<double>(native_value_from(theta));
        push_expected_int(std::sin(theta_dbl));
        push_expected_int(std::cos(theta_dbl));
    }
    EXPECT_VEC_EQ(expected, actual);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
