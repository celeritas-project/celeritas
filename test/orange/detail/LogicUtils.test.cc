//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.test.cc
//---------------------------------------------------------------------------//

#include <string_view>
#include <vector>

#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"
#include "orange/detail/ConvertLogic.hh"
#include "orange/detail/LogicIO.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

OrangeInput
make_input_with_logic(std::vector<logic_int> logic, LogicNotation notation)
{
    OrangeInput input;
    UnitInput unit;
    VolumeInput volume;
    volume.logic = std::move(logic);
    volume.zorder = ZOrder::media;
    unit.volumes.push_back(std::move(volume));
    input.universes.push_back(std::move(unit));
    input.logic = notation;
    input.tol = Tolerance<>::from_default();
    return input;
}

TEST(NotationConverter, basic)
{
    auto round_trip = [](std::string_view postfix, std::string_view infix) {
        auto postfix_expr = string_to_logic(postfix);
        auto infix_expr = convert_to_infix(postfix_expr);
        EXPECT_EQ(logic_to_string(infix_expr), infix);
        auto new_postfix_expr = convert_to_postfix(infix_expr);
        EXPECT_EQ(logic_to_string(new_postfix_expr), postfix);
    };

    round_trip("0 1 ~ & 2 & 3 ~ & 4 & 5 ~ & ~",
               "~ ( 0 & ~ 1 & 2 & ~ 3 & 4 & ~ 5 )");
    round_trip("* ~", "~ *");
    round_trip("0 1 ~ & 2 & * & 3 & 4 ~ & 5 ~ & 6 &",
               "0 & ~ 1 & 2 & * & 3 & ~ 4 & ~ 5 & 6");
    round_trip("0 ~ 1 & 2 & 3 & 7 & 4 5 ~ & 6 ~ & ~ &",
               "~ 0 & 1 & 2 & 3 & 7 & ~ ( 4 & ~ 5 & ~ 6 )");
    round_trip("0 ~ 1 & 2 & 3 & 7 & 4 5 ~ & 6 ~ & ~ |",
               "( ~ 0 & 1 & 2 & 3 & 7 ) | ~ ( 4 & ~ 5 & ~ 6 )");
    round_trip("0 ~ 1 ~ 2 ~ | ~ &", "~ 0 & ~ ( ~ 1 | ~ 2 )");
    round_trip("0 1 ~ & 2 & 3 ~ & 4 & 5 ~ & 6 7 & 8 ~ & 9 & 10 ~ & 11 ~ & ~ &",
               "0 & ~ 1 & 2 & ~ 3 & 4 & ~ 5 & ~ ( 6 & 7 & ~ 8 & 9 & ~ 10 & ~ "
               "11 )");

    round_trip("0 1 ~ & 2 & 3 ~ | 4 & 5 ~ & 6 7 & 8 ~ & 9 & 10 ~ & 11 ~ & ~ &",
               "( ( 0 & ~ 1 & 2 ) | ~ 3 ) & 4 & ~ 5 & ~ ( 6 & 7 & ~ 8 & 9 & ~ "
               "10 & ~ 11 )");

    round_trip("0 1 ~ & 2 & 3 ~ | 4 & 5 ~ & 6 7 & 8 ~ & 9 & 10 ~ & 11 ~ & ~ |",
               "( ( ( 0 & ~ 1 & 2 ) | ~ 3 ) & 4 & ~ 5 ) | ~ ( 6 & 7 & ~ 8 & 9 "
               "& ~ 10 & ~ 11 )");
}

// Test that demorgan's law is applied to input postfix
TEST(NotationConverter, demorgan_postfix_to_infix)
{
    auto input = make_input_with_logic(string_to_logic("0 1 | ~"),
                                       LogicNotation::postfix);
    convert_logic(input, LogicNotation::infix);
    auto& unit = std::get<UnitInput>(input.universes.front());
    auto postfix = convert_to_postfix(unit.volumes.front().logic);
    EXPECT_EQ(logic_to_string(postfix), "0 ~ 1 ~ &");
}

// Transformation is *not* applied if input is infix
TEST(NotationConverter, demorgan_infix_to_infix)
{
    std::vector<logic_int> infix = string_to_logic("~ ( 0 | 1 )");
    auto input = make_input_with_logic(std::move(infix), LogicNotation::infix);
    convert_logic(input, LogicNotation::infix);
    auto& unit = std::get<UnitInput>(input.universes.front());
    auto postfix = convert_to_postfix(unit.volumes.front().logic);
    EXPECT_EQ(logic_to_string(postfix), "0 1 | ~");
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
