//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.test.cc
//---------------------------------------------------------------------------//

#include "orange/detail/LogicUtils.hh"

#include <string>
#include <string_view>
#include <vector>

#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

std::vector<logic_int> postfix_to_infix(std::string_view postfix)
{
    auto postfix_expr = string_to_logic(postfix.data());
    return convert_to_infix(make_span(postfix_expr));
}

std::vector<logic_int> infix_to_postfix(std::vector<logic_int> infix)
{
    return convert_to_postfix(make_span(infix));
}

TEST(NotationConverter, basic)
{
    auto round_trip = [](std::string_view postfix, std::string_view infix) {
        auto infix_expr = postfix_to_infix(postfix);
        EXPECT_EQ(logic_to_string(infix_expr), infix);
        auto postfix_expr = infix_to_postfix(infix_expr);
        EXPECT_EQ(logic_to_string(postfix_expr), postfix);
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
