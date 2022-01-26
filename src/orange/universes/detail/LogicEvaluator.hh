//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LogicEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Span.hh"
#include "orange/Types.hh"
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
    //! Public type aliases
    using SpanConstLogic = Span<const logic_int>;
    using SpanConstSense = Span<const Sense>;
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
} // namespace detail
} // namespace celeritas
