//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilderPolicies.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/VariantUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with postfix operation.
 *
 * This is a policy used as template parameter of the \c
 * LogicBuilder::operator(). The user invokes this class with a node ID
 * (usually representing a cell), and then this class recurses into
 * the daughters using a tree visitor.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "0 1 & 2 & &"}
    all(1, 3, !all(2, 4)) -> {{1, 2, 3, 4}, "0 2 & 1 3 & ~ &"}
 * \endverbatim
 */
class PostfixLogicBuilderPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

    static_assert(std::is_same_v<LocalSurfaceId::size_type, logic_int>,
                  "unsupported: add enum logic conversion for different-sized "
                  "face and surface ints");

  public:
    // Construct with optional mapping and logic vector to append to
    inline PostfixLogicBuilderPolicy(CsgTree const& tree,
                                     VecSurface const* vs,
                                     VecLogic* logic);

    //! Build from a node ID
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

  protected:
    VecLogic& logic() { return *logic_; }
    ContainerVisitor<CsgTree const&, NodeId>& visit() { return visit_node_; }

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_;
    VecLogic* logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with infix operation.
 *
 * This is a policy used as template parameter of the \c
 * LogicBuilder::operator(). The user invokes this class with a node ID
 * (usually representing a cell), and then this class recurses into the
 * daughters using a tree visitor.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "(0 & 1 & 2)"}
    all(1, 3, any(~(2), ~(4))) -> {{1, 2, 3, 4}, "(0 & 2 & (~1 | ~3))"}
 * \endverbatim
 */
class InfixLogicBuilderPolicy : public PostfixLogicBuilderPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using PostfixLogicBuilderPolicy::VecLogic;
    using PostfixLogicBuilderPolicy::VecSurface;
    //!@}

  public:
    using PostfixLogicBuilderPolicy::PostfixLogicBuilderPolicy;

    //! Build from a node ID
    inline void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly
    using PostfixLogicBuilderPolicy::operator();
    // Aliased nodes should never be reachable explicitly
    inline void operator()(Aliased const&);
    // Visit a negated node and append 'not'
    inline void operator()(Negated const&);
    // Visit daughter nodes and append the conjunction.
    inline void operator()(Joined const&);
    //!@}
};

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
/*!
 * Build from a node ID.
 */
void InfixLogicBuilderPolicy::operator()(NodeId const& n)
{
    this->visit()(*this, n);
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
    this->logic().push_back(logic::lnot);
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes and append the conjunction.
 */
void InfixLogicBuilderPolicy::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);
    auto& logic = this->logic();
    logic.push_back(logic::lopen);
    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        logic.push_back(n.op);
        (*this)(*iter++);
    }
    logic.push_back(logic::lclose);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
