//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/LogicEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"

#include "LogicStack.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate a logical expression applied to a vector of senses.
 */
class LogicEvaluator
{
  public:
    //@{
    //! \name Type aliases
    using SpanConstLogic = LdgSpan<logic_int const>;
    using SpanConstSense = Span<Sense const>;
    //@}

  public:
    // Construct with a view to some logic definition
    explicit CELER_FORCEINLINE_FUNCTION LogicEvaluator(SpanConstLogic logic);

    // Evaluate a logical expression, substituting bools from the vector
    inline CELER_FUNCTION bool operator()(SpanConstSense values) const;

  private:
    //// DATA ////

    SpanConstLogic logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with a view to some logic definition.
 */
CELER_FUNCTION LogicEvaluator::LogicEvaluator(SpanConstLogic logic)
    : logic_(logic)
{
    CELER_EXPECT(!logic_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate a logical expression, substituting bools from the sense view.
 */
CELER_FUNCTION bool LogicEvaluator::operator()(SpanConstSense values) const
{
    LogicStack stack;

    for (logic_int lgc : logic_)
    {
        if (!logic::is_operator_token(lgc))
        {
            // Push a boolean from the senses onto the stack
            CELER_EXPECT(lgc < values.size());
            stack.push(static_cast<bool>(values[lgc]));
            continue;
        }

        // Apply logic operator
        switch (lgc)
        {
            // clang-format off
            case logic::ltrue: stack.push(true);  break;
            case logic::lor:   stack.apply_or();  break;
            case logic::land:  stack.apply_and(); break;
            case logic::lnot:  stack.apply_not(); break;
            default:           CELER_ASSERT_UNREACHABLE();
        }
        // clang-format on
    }
    CELER_ENSURE(stack.size() == 1);
    return stack.top();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
