//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/InfixEvaluator.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/InfixEvaluator.hh"

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

constexpr auto s_in = Sense::inside;
constexpr auto s_out = Sense::outside;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(InfixEvaluatorTest, evaluate)
{
    // Logic for alpha : !1 | 2 | !3 | 4 | !8
    // With senses substituted: F | F | F | F | F
    auto const alpha_logic = string_to_logic("~1 | 2 |~3 | 4 | ~8");

    //
    // Logic for beta : ((((5 & !1) & 6) & !7) & 8)
    // With senses substituted: ((((T & F) & F) & T) & T)
    auto const beta_logic = string_to_logic("((((5 & ~1) & 6) & ~7) & 8)");

    // Logic for gamma : 8 ~ ~ ~ ~
    // With senses substituted: T
    auto const gamma_logic = string_to_logic("8");

    // Logic for delta : ((((!1 | 2 | !3 | 4) & !5 | 1 | !6 | 7) & 8) & !0)
    // With senses substituted: ((((F | F | F | T) & F | 1 | F | F) & T) & T)
    auto const delta_logic = string_to_logic(
        "(((( ~1 | 2 | ~3 | 4) & ~5 | 1 | ~6 | 7) & 8) & ~0)");

    auto const everywhere_logic = string_to_logic("*");

    //// CREATE ////

    InfixEvaluator eval_alpha(make_span(alpha_logic));
    InfixEvaluator eval_beta(make_span(beta_logic));
    InfixEvaluator eval_gamma(make_span(gamma_logic));
    InfixEvaluator eval_delta(make_span(delta_logic));
    InfixEvaluator eval_everywhere(make_span(everywhere_logic));

    //// EVALUATE ////

    VecSense senses
        = {s_in, s_out, s_in, s_out, s_in, s_out, s_in, s_in, s_out};

    auto eval = [&senses](FaceId i) {
        CELER_EXPECT(i < senses.size());
        return static_cast<bool>(senses[i.unchecked_get()]);
    };
    EXPECT_FALSE(eval_alpha(eval));
    EXPECT_FALSE(eval_beta(eval));
    EXPECT_TRUE(eval_gamma(eval));
    EXPECT_TRUE(eval_everywhere(eval));

    // Should evaluate to true (inside delta)
    senses
        = {s_in, s_out, s_in, s_out, s_out, s_out, s_out, s_in, s_out, s_out};
    EXPECT_TRUE(eval_delta(eval));
    EXPECT_TRUE(eval_everywhere(eval));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
