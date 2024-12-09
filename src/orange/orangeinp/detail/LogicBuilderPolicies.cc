//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilderPolicies.cc
//---------------------------------------------------------------------------//
#include "LogicBuilderPolicies.hh"

#include <vector>

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to the logic expression.
 *
 * The surface mapping vector is *optional*.
 */
InfixLogicBuilderPolicy::InfixLogicBuilderPolicy(CsgTree const& tree,
                                                 VecSurface const* vs,
                                                 VecLogic* logic)
    : visit_node_{tree}, mapping_{vs}, logic_{logic}
{
    CELER_EXPECT(logic_);
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
void InfixLogicBuilderPolicy::operator()(NodeId const& n)
{
    visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
void InfixLogicBuilderPolicy::operator()(True const&)
{
    logic_->push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
void InfixLogicBuilderPolicy::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
void InfixLogicBuilderPolicy::operator()(Surface const& s)
{
    CELER_EXPECT(s.id < logic::lbegin);
    // Get index of original surface or remapped
    logic_int sidx = [this, sid = s.id] {
        if (!mapping_)
        {
            return sid.unchecked_get();
        }
        else
        {
            // Remap by finding position of surface in our mapping
            auto iter = find_sorted(mapping_->begin(), mapping_->end(), sid);
            CELER_ASSERT(iter != mapping_->end());
            return logic_int(iter - mapping_->begin());
        }
    }();

    logic_->push_back(sidx);
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * Aliased node shouldn't be reachable if the tree is fully simplified, but
 * could be reachable for testing purposes.
 */
void InfixLogicBuilderPolicy::operator()(Aliased const& n)
{
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit a negated node and append 'not'.
 */
void InfixLogicBuilderPolicy::operator()(Negated const& n)
{
    logic_->push_back(logic::lnot);
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes and append the conjunction.
 */
void InfixLogicBuilderPolicy::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);
    logic_->push_back(logic::lopen);
    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        logic_->push_back(n.op);
        (*this)(*iter++);
    }
    logic_->push_back(logic::lclose);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to the logic expression.
 *
 * The surface mapping vector is *optional*.
 */
PostfixLogicBuilderPolicy::PostfixLogicBuilderPolicy(CsgTree const& tree,
                                                     VecSurface const* vs,
                                                     VecLogic* logic)
    : visit_node_{tree}, mapping_{vs}, logic_{logic}
{
    CELER_EXPECT(logic_);
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
void PostfixLogicBuilderPolicy::operator()(NodeId const& n)
{
    visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
void PostfixLogicBuilderPolicy::operator()(True const&)
{
    logic_->push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
void PostfixLogicBuilderPolicy::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
void PostfixLogicBuilderPolicy::operator()(Surface const& s)
{
    CELER_EXPECT(s.id < logic::lbegin);
    // Get index of original surface or remapped
    logic_int sidx = [this, sid = s.id] {
        if (!mapping_)
        {
            return sid.unchecked_get();
        }
        else
        {
            // Remap by finding position of surface in our mapping
            auto iter = find_sorted(mapping_->begin(), mapping_->end(), sid);
            CELER_ASSERT(iter != mapping_->end());
            return logic_int(iter - mapping_->begin());
        }
    }();

    logic_->push_back(sidx);
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * Aliased node shouldn't be reachable if the tree is fully simplified, but
 * could be reachable for testing purposes.
 */
void PostfixLogicBuilderPolicy::operator()(Aliased const& n)
{
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit a negated node and append 'not'.
 */
void PostfixLogicBuilderPolicy::operator()(Negated const& n)
{
    (*this)(n.node);
    logic_->push_back(logic::lnot);
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes and append the conjunction.
 */
void PostfixLogicBuilderPolicy::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);

    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        (*this)(*iter++);
        logic_->push_back(n.op);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
