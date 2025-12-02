//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/LogicEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/LdgIterator.hh"
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

    // Evaluate a logical expression, with on-the-fly sense evaluation
    template<class F, std::enable_if_t<std::is_invocable_v<F, FaceId>, bool> = true>
    inline CELER_FUNCTION bool operator()(F&& eval_sense) const;

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
    auto calc_sense
        = [&](FaceId face_id) -> Sense { return values[face_id.get()]; };
    return (*this)(calc_sense);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate a logical expression, with on-the-fly sense evaluation.
 */
template<class F, std::enable_if_t<std::is_invocable_v<F, FaceId>, bool>>
CELER_FUNCTION bool LogicEvaluator::operator()(F&& eval_sense) const
{
    LogicStack stack;

    for (logic_int lgc : logic_)
    {
        if (!logic::is_operator_token(lgc))
        {
            // Push a boolean from the senses onto the stack
            stack.push(static_cast<bool>(eval_sense(FaceId{lgc})));
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

//! Alias explicitly naming the notation used
using PostfixEvaluator = LogicEvaluator;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
