//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    if (on_surface_)
    {
        // Clear on-surface value
        *on_surface_ = {};
    }
    return visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate a surface.
 */
SignedSense SenseEvaluator::operator()(Surface const& s) const
{
    CELER_EXPECT(s.id < surfaces_.size());

    auto result = std::visit(
        [&pos = this->pos_](auto const& surf) { return surf.calc_sense(pos); },
        surfaces_[s.id.get()]);

    if (result == SignedSense::on && on_surface_)
    {
        *on_surface_ = s.id;
        return result;
    }

    /*!
     * \todo "inside" wrt a surface (i.e. negative quadric) is "false", so we
     * have to flip the result. We should change the value of \c Sense::inside
     * to \c true.
     */
    static_assert(Sense::inside == to_sense(false));
    return flip_sense(result);
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
