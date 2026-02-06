//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/LogicEvaluator.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/LogicEvaluator.hh"

#include "orange/detail/LogicIO.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

using VecSense = std::vector<Sense>;

constexpr auto lbegin = logic::lbegin;
constexpr auto ltrue = logic::ltrue;
constexpr auto lor = logic::lor;
constexpr auto land = logic::land;
constexpr auto lnot = logic::lnot;

constexpr auto s_in = Sense::inside;
constexpr auto s_out = Sense::outside;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(LogicEvaluatorTest, enumeration)
{
    EXPECT_GE(logic::lclose, lbegin);
    EXPECT_GE(logic::lopen, lbegin);
    EXPECT_GE(ltrue, lbegin);
    EXPECT_GE(lnot, lbegin);
    EXPECT_GE(land, lbegin);
    EXPECT_GE(lor, lbegin);
    EXPECT_EQ(ltrue, static_cast<logic_int>(-1));

    EXPECT_EQ('*', to_char(ltrue));
    EXPECT_EQ('|', to_char(lor));
    EXPECT_EQ('&', to_char(land));
    EXPECT_EQ('~', to_char(lnot));
}

TEST(LogicEvaluatorTest, evaluate)
{
    // Logic for alpha : 1 2 ~ & 3 & 4 ~ & ~ ~ 8 ~ ~ & ~
    // With senses substituted: T F ~ & T & F ~ & T & ~
    auto const alpha_logic
        = string_to_logic("1 2 ~ & 3 & 4 ~ & ~ ~ 8 ~ ~ & ~");

    // Logic for beta : 5 1 ~ & 6 & 7 ~ & ~ ~ 8 ~ ~ & ~
    // With senses substituted: T T ~ & F & F ~ & T & ~
    auto const beta_logic = string_to_logic("5 1 ~ & 6 & 7 ~ & ~ ~ 8 ~ ~ & ~");

    // Logic for gamma : 8 ~ ~ ~ ~
    // With senses substituted: T
    auto const gamma_logic = string_to_logic("8");

    // 1 2 ~ & 3 & 4 ~ & ~ 5 1 ~ & 6 & 7 ~ & ~ & 8 & 0 ~ &
    auto const delta_logic = string_to_logic(
        "1 2 ~ & 3 & 4 ~ & ~ 5 1 ~ & 6 & 7 ~ & ~ & 8 & 0 ~ &");

    auto const everywhere_logic = string_to_logic("*");

    //// CREATE ////

    LogicEvaluator eval_alpha(make_span(alpha_logic));
    LogicEvaluator eval_beta(make_span(beta_logic));
    LogicEvaluator eval_gamma(make_span(gamma_logic));
    LogicEvaluator eval_delta(make_span(delta_logic));
    LogicEvaluator eval_everywhere(make_span(everywhere_logic));

    //// EVALUATE ////

    VecSense senses
        = {s_in, s_out, s_in, s_out, s_in, s_out, s_in, s_in, s_out};
    EXPECT_FALSE(eval_alpha(make_span(senses)));
    EXPECT_TRUE(eval_beta(make_span(senses)));
    EXPECT_TRUE(eval_gamma(make_span(senses)));
    EXPECT_TRUE(eval_everywhere(make_span(senses)));

    // Should evaluate to true (inside delta)
    senses
        = {s_in, s_out, s_in, s_out, s_out, s_out, s_out, s_in, s_out, s_out};
    EXPECT_TRUE(eval_delta(make_span(senses)));
    EXPECT_TRUE(eval_everywhere(make_span(senses)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
