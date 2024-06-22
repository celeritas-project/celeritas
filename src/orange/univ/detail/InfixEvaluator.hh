//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/InfixEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate an explicit infix logical expression applied to a vector of senses.
 */
class InfixEvaluator
{
  public:
    //@{
    //! \name Type aliases
    using SpanConstLogic = LdgSpan<logic_int const>;
    using SpanConstSense = Span<Sense const>;
    //@}

  public:
    // Construct with a view to some logic definition
    explicit CELER_FORCEINLINE_FUNCTION InfixEvaluator(SpanConstLogic logic);

    // Evaluate a logical expression, substituting bools from the vector
    inline CELER_FUNCTION bool operator()(SpanConstSense values) const;

  private:
    // Short-circuit evaluation of the second operand
    inline CELER_FUNCTION uint32_t short_circuit(uint32_t i) const;

    //// DATA ////

    SpanConstLogic logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with a view to some logic definition.
 */
CELER_FUNCTION InfixEvaluator::InfixEvaluator(SpanConstLogic logic)
    : logic_(logic)
{
    CELER_EXPECT(!logic_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate a logical expression, substituting bools from the vector.
 * TODO: values don't need to be pre-calculated
 */
CELER_FUNCTION bool InfixEvaluator::operator()(SpanConstSense values) const
{
    bool result{true};

    for (auto par_depth{0}, i{0}; i < logic_.size(); ++i)
    {
        if (auto const lgc{logic_[i]}; !logic::is_operator_token(lgc))
        {
            CELER_EXPECT(lgc < values.size());
            result = static_cast<bool>(values[lgc]);
        }
        else if ((lgc == logic::lor && result)
                 || (lgc == logic::land && !result))
        {
            if (par_depth == 0)
            {
                break;
            }
            --par_depth;
            i = short_circuit(i);
        }
        else
        {
            switch (lgc)
            {
                case logic::lnot: {
                    auto next = logic_[++i];
                    // negation of a sub-expression is not supported
                    CELER_EXPECT(!logic::is_operator_token(next));
                    result = !static_cast<bool>(values[next]);
                    break;
                }
                    // clang-format off
                case logic::ltrue: result = true; break;
                case logic::lpar_open: ++par_depth; break;
                case logic::lpar_close: --par_depth; break;
                case logic::land: break;
                case logic::lor: break;
                default: CELER_ASSERT_UNREACHABLE();
                    // clang-format on
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Short-circuit evaluation of the second operand.
 */
CELER_FUNCTION uint32_t InfixEvaluator::short_circuit(uint32_t i) const
{
    for (auto parenthesis_depth{1}; parenthesis_depth > 0;)
    {
        if (auto const lgc{logic_[++i]}; lgc == logic::lpar_open)
        {
            ++parenthesis_depth;
        }
        else if (lgc == logic::lpar_close)
        {
            --parenthesis_depth;
        }
    }
    return i;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
