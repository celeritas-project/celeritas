//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/universes/detail/LogicEvaluator.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/LogicEvaluator.hh"

#include <iomanip>

#include "celeritas_test.hh"

using celeritas::detail::LogicEvaluator;

using celeritas::make_span;
using celeritas::detail::LogicEvaluator;

using VecSense = std::vector<celeritas::Sense>;

using logic_int       = celeritas::logic_int;
constexpr auto lbegin = celeritas::logic::lbegin;
constexpr auto ltrue  = celeritas::logic::ltrue;
constexpr auto lor    = celeritas::logic::lor;
constexpr auto land   = celeritas::logic::land;
constexpr auto lnot   = celeritas::logic::lnot;
constexpr auto lend   = celeritas::logic::lend;

constexpr auto s_in  = celeritas::Sense::inside;
constexpr auto s_out = celeritas::Sense::outside;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(LogicEvaluatorTest, enumeration)
{
    EXPECT_GE(ltrue, lbegin);
    EXPECT_GE(lnot, lbegin);
    EXPECT_GE(land, lbegin);
    EXPECT_GE(lor, lbegin);
    EXPECT_LT(lbegin, lend);

    EXPECT_EQ('*', to_char(ltrue));
    EXPECT_EQ('|', to_char(lor));
    EXPECT_EQ('&', to_char(land));
    EXPECT_EQ('~', to_char(lnot));
}

TEST(LogicEvaluatorTest, evaluate)
{
    // Logic for alpha : 1 2 ~ & 3 & 4 ~ & ~ ~ 8 ~ ~ & ~
    // With senses substituted: T F ~ & T & F ~ & T & ~
    const logic_int alpha_logic[] = {1,
                                     2,
                                     lnot,
                                     land,
                                     3,
                                     land,
                                     4,
                                     lnot,
                                     land,
                                     lnot,
                                     lnot,
                                     8,
                                     lnot,
                                     lnot,
                                     land,
                                     lnot};

    // Logic for beta : 5 1 ~ & 6 & 7 ~ & ~ ~ 8 ~ ~ & ~
    // With senses substituted: T T ~ & F & F ~ & T & ~
    const logic_int beta_logic[] = {5,
                                    1,
                                    lnot,
                                    land,
                                    6,
                                    land,
                                    7,
                                    lnot,
                                    land,
                                    lnot,
                                    lnot,
                                    8,
                                    lnot,
                                    lnot,
                                    land,
                                    lnot};

    // Logic for gamma : 8 ~ ~ ~ ~
    // With senses substituted: T
    const logic_int gamma_logic[] = {8};

    // 1 2 ~ & 3 & 4 ~ & ~ 5 1 ~ & 6 & 7 ~ & ~ & 8 & 0 ~ &
    const logic_int delta_logic[] = {1,    2,    lnot, land, 3,    land, 4,
                                     lnot, land, lnot, 5,    1,    lnot, land,
                                     6,    land, 7,    lnot, land, lnot, land,
                                     8,    land, 0,    lnot, land};

    const logic_int everywhere_logic[] = {ltrue};

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
