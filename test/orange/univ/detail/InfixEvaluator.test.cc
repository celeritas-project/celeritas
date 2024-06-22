//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/InfixEvaluator.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/InfixEvaluator.hh"

#include <iomanip>

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
constexpr auto lpar_open = logic::lpar_open;
constexpr auto lpar_close = logic::lpar_close;
constexpr auto lend = logic::lend;

constexpr auto s_in = Sense::inside;
constexpr auto s_out = Sense::outside;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(InfixEvaluatorTest, enumeration)
{
    EXPECT_GE(ltrue, lbegin);
    EXPECT_GE(lnot, lbegin);
    EXPECT_GE(land, lbegin);
    EXPECT_GE(lor, lbegin);
    EXPECT_GE(lpar_open, lbegin);
    EXPECT_GE(lpar_close, lbegin);
    EXPECT_LT(lbegin, lend);

    EXPECT_EQ('*', to_char(ltrue));
    EXPECT_EQ('|', to_char(lor));
    EXPECT_EQ('&', to_char(land));
    EXPECT_EQ('~', to_char(lnot));
    EXPECT_EQ('(', to_char(lpar_open));
    EXPECT_EQ(')', to_char(lpar_close));
}

TEST(InfixEvaluatorTest, evaluate)
{
    // Logic for alpha : !1 | 2 | !3 | 4 | !8
    // With senses substituted: F | F | F | F | F
    logic_int const alpha_logic[]
        = {lnot, 1, lor, 2, lor, lnot, 3, lor, 4, lor, lnot, 8};

    //
    // Logic for beta : ((((5 & !1) & 6) & !7) & 8)
    // With senses substituted: ((((T & F) & F) & T) & T)
    logic_int const beta_logic[] = {lpar_open,
                                    lpar_open,
                                    lpar_open,
                                    lpar_open,
                                    5,
                                    land,
                                    lnot,
                                    1,
                                    lpar_close,
                                    land,
                                    6,
                                    lpar_close,
                                    land,
                                    lnot,
                                    7,
                                    lpar_close,
                                    land,
                                    8,
                                    lpar_close};

    // Logic for gamma : 8 ~ ~ ~ ~
    // With senses substituted: T
    logic_int const gamma_logic[] = {8};

    // Logic for delta : ((((!1 | 2 | !3 | 4) & !5 | 1 | !6 | 7) & 8) & !0)
    // With senses substituted: ((((F | F | F | T) & F | 1 | F | F) & T) & T)
    logic_int const delta_logic[]
        = {lpar_open, lpar_open, lpar_open, lpar_open,  lnot, 1,   lor,
           2,         lor,       lnot,      3,          lor,  4,   lpar_close,
           land,      lnot,      5,         lor,        1,    lor, lnot,
           6,         lor,       7,         lpar_close, land, 8,   lpar_close,
           land,      lnot,      0,         lpar_close};

    logic_int const everywhere_logic[] = {ltrue};

    //// CREATE ////

    InfixEvaluator eval_alpha(make_span(alpha_logic));
    InfixEvaluator eval_beta(make_span(beta_logic));
    InfixEvaluator eval_gamma(make_span(gamma_logic));
    InfixEvaluator eval_delta(make_span(delta_logic));
    InfixEvaluator eval_everywhere(make_span(everywhere_logic));

    //// EVALUATE ////

    VecSense senses
        = {s_in, s_out, s_in, s_out, s_in, s_out, s_in, s_in, s_out};
    EXPECT_FALSE(eval_alpha(make_span(senses)));
    EXPECT_FALSE(eval_beta(make_span(senses)));
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
