//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/InfixStringBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

#include "corecel/cont/VariantUtils.hh"
#include "orange/SenseUtils.hh"

#include "../CsgTree.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Transform a CSG node into a string expression.
 *
 * The string will be a combination of:
 * - an \c any function for a union of all listed components
 * - an \c all function for an intersection of all listed components
 * - a \c ! negation operator applied to the left of an operation or other
 *   negation
 * - a surface ID preceded by a \c - or \c + indicating "inside" or "outside",
 *   respectively.
 *
 * Example of a cylindrical shell: \verbatim
   all(all(+0, -1, -3), !all(+0, -1, -2))
 * \endverbatim
 */
class InfixStringBuilder
{
  public:
    // Construct with tree and a stream to write to
    inline InfixStringBuilder(CsgTree const& tree, std::ostream* os);

    // Build from a node ID
    inline void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly

    // Append 'true'
    inline void operator()(True const&);
    // False is never explicitly part of the node tree
    inline void operator()(False const&);
    // Append a surface ID
    inline void operator()(Surface const&);
    // Aliased nodes should never be reachable explicitly
    inline void operator()(Aliased const&);
    // Visit a negated node and append 'not'
    inline void operator()(Negated const&);
    // Visit daughter nodes and append the conjunction.
    inline void operator()(Joined const&);
    //!@}

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    std::ostream* os_;
    bool negated_{false};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to an output stream.
 */
InfixStringBuilder::InfixStringBuilder(CsgTree const& tree, std::ostream* os)
    : visit_node_{tree}, os_{os}
{
    CELER_EXPECT(os_);
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
void InfixStringBuilder::operator()(NodeId const& n)
{
    visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
void InfixStringBuilder::operator()(True const&)
{
    *os_ << (negated_ ? 'F' : 'T');
    negated_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
void InfixStringBuilder::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
void InfixStringBuilder::operator()(Surface const& s)
{
    CELER_EXPECT(s.id < logic::lbegin);

    static_assert(to_sense(true) == Sense::outside);
    *os_ << (negated_ ? '-' : '+') << s.id.unchecked_get();
    negated_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * Note: aliased node won't be reachable if a tree is fully simplified, *but* a
 * node can be printed for testing before it's simplified.
 */
void InfixStringBuilder::operator()(Aliased const& n)
{
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit a negated node and append 'not'.
 */
void InfixStringBuilder::operator()(Negated const& n)
{
    if (negated_)
    {
        // Note: this won't happen for simplified expressions but can be for
        // testing unsimplified expressions.
        *os_ << '!';
    }
    negated_ = true;
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes and append the conjunction.
 */
void InfixStringBuilder::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);

    if (negated_)
    {
        *os_ << '!';
    }
    negated_ = false;
    *os_ << (n.op == op_and ? "all" : n.op == op_or ? "any" : "XXX") << '(';

    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        *os_ << ", ";
        (*this)(*iter++);
    }
    *os_ << ')';
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
