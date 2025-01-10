//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/InfixEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/LdgIterator.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate an explicit infix logical expression applied to a vector of senses.
 *
 * Explicit infix notation explicitly spells out the intersection operator,
 * i.e., \c logic::land wehereas implicit infix notation omits it.
 *
 * These two expressions are equivalent:
 * Implicit notation: A(BC | DE)
 * Explicit notation: A&((B&C) | (D&E))
 */
class InfixEvaluator
{
  public:
    //@{
    //! \name Type aliases
    using SpanConstLogic = LdgSpan<logic_int const>;
    //@}

  public:
    // Construct with a view to some logic definition
    explicit CELER_FORCEINLINE_FUNCTION InfixEvaluator(SpanConstLogic logic);

    // Evaluate a logical expression, substituting bools from the vector
    template<class F>
    inline CELER_FUNCTION bool operator()(F&& eval_sense) const;

  private:
    // Short-circuit evaluation of the second operand
    inline CELER_FUNCTION size_type short_circuit(size_type i) const;

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
 *
 * The functor eval_sense takes as argument a \c celeritas::FaceId present in
 * the logic expression passed to the constructor and returns a boolean value
 * matching \c celeritas::Sense
 */
template<class F>
CELER_FUNCTION bool InfixEvaluator::operator()(F&& eval_sense) const
{
    bool result{true};

    int par_depth{0};
    size_type i{0};
    while (i < logic_.size())
    {
        if (logic_int const lgc{logic_[i]}; !logic::is_operator_token(lgc))
        {
            result = static_cast<bool>(eval_sense(FaceId{lgc}));
        }
        else if ((lgc == logic::lor && result)
                 || (lgc == logic::land && !result))
        {
            if (par_depth == 0)
            {
                break;
            }
            --par_depth;
            i = this->short_circuit(i);
        }
        else if (lgc == logic::ltrue)
        {
            result = true;
        }
        else if (lgc == logic::lopen)
        {
            ++par_depth;
        }
        else if (lgc == logic::lclose)
        {
            CELER_ASSERT(par_depth > 0);
            --par_depth;
        }
        else if (lgc == logic::lnot)
        {
            // negation of a sub-expression is not supported
            CELER_ASSUME(i + 1 < logic_.size());
            CELER_EXPECT(!logic::is_operator_token(logic_[i + 1]));
            result = !static_cast<bool>(eval_sense(FaceId{logic_[++i]}));
        }
        ++i;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Short-circuit evaluation of the second operand.
 */
CELER_FUNCTION size_type InfixEvaluator::short_circuit(size_type i) const
{
    int par_depth{1};
    while (par_depth > 0)
    {
        CELER_ASSUME(i + 1 < logic_.size());
        if (logic_int lgc = logic_[++i]; lgc == logic::lopen)
        {
            ++par_depth;
        }
        else if (lgc == logic::lclose)
        {
            --par_depth;
        }
    }
    return i;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
