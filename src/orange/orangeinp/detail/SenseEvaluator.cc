//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SenseEvaluator.cc
//---------------------------------------------------------------------------//
#include "SenseEvaluator.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate from a node ID.
 */
SignedSense SenseEvaluator::operator()(NodeId const& n) const
{
    return visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate a surface.
 */
SignedSense SenseEvaluator::operator()(Surface const& s) const
{
    CELER_EXPECT(s.id < surfaces_.size());

    return std::visit(
        [&pos = this->pos_](auto const& surf) { return surf.calc_sense(pos); },
        surfaces_[s.id.get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Redirect to an aliased node.
 */
SignedSense SenseEvaluator::operator()(Aliased const& n) const
{
    return (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Negate the result of a node.
 */
SignedSense SenseEvaluator::operator()(Negated const& n) const
{
    return flip_sense((*this)(n.node));
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes to evaluate the combined sense.
 */
SignedSense SenseEvaluator::operator()(Joined const& j) const
{
    CELER_ASSUME(j.op == op_and || j.op == op_or);

    // Only keep testing if this sense results:
    // short circuit for the other sense, short circuit for being *on* a
    // surface too
    auto const maybe
        = (j.op == op_and ? SignedSense::inside : SignedSense::outside);

    SignedSense result{};

    for (NodeId const& d : j.nodes)
    {
        result = (*this)(d);
        if (result != maybe)
        {
            break;
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
