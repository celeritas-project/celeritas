//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilderPolicies.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
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
 * Base class for logic builder policies following CRTP pattern.
 *
 * The call operator for Negated and Joined are not implemented in the base
 * policy and must be provided by the derived class.
 */
template<class BuilderPolicy>
class BaseLogicBuilderPolicy
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
    // Construct with optional mapping
    inline BaseLogicBuilderPolicy(CsgTree const& tree, VecSurface const* vs);

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
    //!@}

    //! Access the logic expression
    VecLogic& logic() { return logic_; }

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_;
    VecLogic logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with optional mapping.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 */
template<class BuilderPolicy>
BaseLogicBuilderPolicy<BuilderPolicy>::BaseLogicBuilderPolicy(
    CsgTree const& tree, VecSurface const* vs)
    : visit_node_{tree}, mapping_{vs}
{
    static_assert(std::is_base_of_v<BaseLogicBuilderPolicy, BuilderPolicy>,
                  "CRTP: template parameter must be derived class");
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
template<class BuilderPolicy>
void BaseLogicBuilderPolicy<BuilderPolicy>::operator()(NodeId const& n)
{
    visit_node_(static_cast<BuilderPolicy&>(*this), n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
template<class BuilderPolicy>
void BaseLogicBuilderPolicy<BuilderPolicy>::operator()(True const&)
{
    logic_.push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
template<class BuilderPolicy>
void BaseLogicBuilderPolicy<BuilderPolicy>::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
template<class BuilderPolicy>
void BaseLogicBuilderPolicy<BuilderPolicy>::operator()(Surface const& s)
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

    logic_.push_back(sidx);
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * Aliased node shouldn't be reachable if the tree is fully simplified, but
 * could be reachable for testing purposes.
 */
template<class BuilderPolicy>
void BaseLogicBuilderPolicy<BuilderPolicy>::operator()(Aliased const& n)
{
    (*this)(n.node);
}

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
    : public BaseLogicBuilderPolicy<PostfixLogicBuilderPolicy>
{
  public:
    //!@{
    //! \name Type aliases
    using BaseLogicBuilderPolicy::VecLogic;
    using BaseLogicBuilderPolicy::VecSurface;
    //!@}

  public:
    using BaseLogicBuilderPolicy::BaseLogicBuilderPolicy;

    //!@{
    //! \name Visit a node directly
    using BaseLogicBuilderPolicy::operator();
    // Visit a negated node and append 'not'
    void operator()(Negated const& n)
    {
        (*this)(n.node);
        logic().push_back(logic::lnot);
    }
    // Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n)
    {
        CELER_EXPECT(n.nodes.size() > 1);

        // Visit first node, then add conjunction for subsequent nodes
        auto iter = n.nodes.begin();
        (*this)(*iter++);

        while (iter != n.nodes.end())
        {
            (*this)(*iter++);
            logic().push_back(n.op);
        }
    }
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
class InfixLogicBuilderPolicy
    : public BaseLogicBuilderPolicy<InfixLogicBuilderPolicy>
{
  public:
    //!@{
    //! \name Type aliases
    using BaseLogicBuilderPolicy::VecLogic;
    using BaseLogicBuilderPolicy::VecSurface;
    //!@}

  public:
    using BaseLogicBuilderPolicy::BaseLogicBuilderPolicy;

    //!@{
    //! \name Visit a node directly
    using BaseLogicBuilderPolicy::operator();
    //! Append 'not' and visit a negated node
    void operator()(Negated const& n)
    {
        this->logic().push_back(logic::lnot);
        (*this)(n.node);
    }

    //! Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n)
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
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
