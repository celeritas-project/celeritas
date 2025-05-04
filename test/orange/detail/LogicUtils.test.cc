//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.test.cc
//---------------------------------------------------------------------------//

#include "orange/detail/LogicUtils.hh"

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

TEST(NotationConverter, basic)
{
    {
        auto infix_expr = postfix_to_infix("0 1 ~ & 2 & 3 ~ & 4 & 5 ~ & ~");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "~ ( 0 & ~ 1 & 2 & ~ 3 & 4 & ~ 5 )");
    }
    {
        auto infix_expr = postfix_to_infix("* ~");
        EXPECT_EQ(logic_to_string(infix_expr), "~ *");
    }
    {
        auto infix_expr
            = postfix_to_infix("0 1 ~ & 2 & * & 3 & 4 ~ & 5 ~ & 6 &");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "0 & ~ 1 & 2 & * & 3 & ~ 4 & ~ 5 & 6");
    }
    {
        auto infix_expr
            = postfix_to_infix("0 ~ 1 & 2 & 3 & 7 & 4 5 ~ & 6 ~ & ~ &");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "~ 0 & 1 & 2 & 3 & 7 & ~ ( 4 & ~ 5 & ~ 6 )");
    }
    {
        auto infix_expr
            = postfix_to_infix("0 ~ 1 & 2 & 3 & 7 & 4 5 ~ & 6 ~ & ~ |");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "( ~ 0 & 1 & 2 & 3 & 7 ) | ~ ( 4 & ~ 5 & ~ 6 )");
    }
    {
        auto infix_expr = postfix_to_infix("0 ~ 1 ~ 2 ~ | ~ &");
        EXPECT_EQ(logic_to_string(infix_expr), "~ 0 & ~ ( ~ 1 | ~ 2 )");
    }
    {
        auto infix_expr = postfix_to_infix(
            "0 1 ~ & 2 & 3 ~ & 4 & 5 ~ & 6 7 & 8 ~ & 9 & 10 ~ & 11 ~ & ~ &");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "0 & ~ 1 & 2 & ~ 3 & 4 & ~ 5 & ~ ( 6 & 7 & ~ 8 & 9 & ~ 10 & "
                  "~ 11 )");
    }
    {
        auto infix_expr = postfix_to_infix(
            "0 1 ~ & 2 & 3 ~ | 4 & 5 ~ & 6 7 & 8 ~ & 9 & 10 ~ & 11 ~ & ~ &");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "( ( 0 & ~ 1 & 2 ) | ~ 3 ) & 4 & ~ 5 & ~ ( 6 & 7 & ~ 8 & 9 "
                  "& ~ 10 & ~ 11 )");
    }
    {
        auto infix_expr = postfix_to_infix(
            "0 1 ~ & 2 & 3 ~ | 4 & 5 ~ & 6 7 & 8 ~ & 9 & 10 ~ & 11 ~ & ~ |");
        EXPECT_EQ(logic_to_string(infix_expr),
                  "( ( ( 0 & ~ 1 & 2 ) | ~ 3 ) & 4 & ~ 5 ) | ~ ( 6 & 7 & ~ 8 "
                  "& 9 & ~ 10 & ~ 11 )");
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas