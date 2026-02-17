//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildLogic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <type_traits>
#include <vector>

#include "corecel/cont/VariantUtils.hh"
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
 * Result of building a logic representation of a node.
 */
struct BuildLogicResult
{
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;

    VecSurface faces;
    VecLogic logic;
};

//---------------------------------------------------------------------------//
/*!
 * Construct a logic representation of a node.
 *
 * The result is a pair of vectors: the sorted surface IDs comprising the faces
 * of this volume, and the logical representation using \em face IDs, i.e. with
 * the surfaces remapped to the index of the surface in the face vector.
 *
 * The function is templated on a policy class that determines the logic
 * representation. The policy acts as a factory that creates a visitor to build
 * the logic expression.
 *
 * The per-node local surfaces (faces) are sorted in ascending order of ID, not
 * of access, since they're always evaluated sequentially rather than as part
 * of the logic evaluation itself.
 */
template<class LogicPolicy>
BuildLogicResult build_logic(LogicPolicy const& policy, NodeId n);

//---------------------------------------------------------------------------//
// BUILDERS
//---------------------------------------------------------------------------//
/*!
 * Base class for logic builder visitors following CRTP pattern.
 *
 * Visitors recursively traverse the CSG tree and append to a logic vector.
 * The call operator for Negated and Joined are not implemented in the base
 * visitor and must be provided by the derived class.
 */
template<class VisitorImpl>
class BaseLogicBuilder
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
    // Construct with optional mapping pointer
    BaseLogicBuilder(CsgTree const& tree,
                     VecLogic& logic,
                     VecSurface const* vs = nullptr);
    //! Build from a node ID
    void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly
    // Append 'true'
    void operator()(True const&);
    // False is never explicitly part of the node tree
    void operator()(False const&);
    // Append a surface ID
    void operator()(Surface const&);
    // Aliased nodes should never be reachable explicitly
    void operator()(Aliased const&);
    //!@}

  protected:
    //! Append a logic token
    void push_back(logic_int lgc) { logic_.push_back(lgc); }

  private:
    VecLogic& logic_;
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * Visitor for constructing logic in postfix notation.
 *
 * Example: \verbatim
    all(1, 3, 5) -> "0 1 & 2 & &"
    all(1, 3, !all(2, 4)) -> "0 2 & 1 3 & ~ &"
 * \endverbatim
 */
class PostfixLogicBuilder : public BaseLogicBuilder<PostfixLogicBuilder>
{
  public:
    using BaseLogicBuilder::BaseLogicBuilder;

    //!@{
    //! \name Visit a node directly
    using BaseLogicBuilder::operator();

    // Visit a negated node and append 'not'.
    void operator()(Negated const& n);

    // Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Visitor for constructing logic in infix notation.
 *
 * Example: \verbatim
    all(1, 3, 5) -> "(0 & 1 & 2)"
    all(1, 3, any(~(2), ~(4))) -> "0 & 2 & (~1 | ~3)"
 * \endverbatim
 */
class InfixLogicBuilder : public BaseLogicBuilder<InfixLogicBuilder>
{
  public:
    using BaseLogicBuilder::BaseLogicBuilder;

    //!@{
    //! \name Visit a node directly
    using BaseLogicBuilder::operator();

    // Append 'not' and visit a negated node.
    void operator()(Negated const& n);

    // Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n);
    //!@}

  private:
    int depth_{0};
};

//---------------------------------------------------------------------------//
// POLICIES
//---------------------------------------------------------------------------//
/*!
 * Policy for building logic expressions.
 * \tparam LogicBuilder Visitor that constructs an input logic vector.
 *
 * This immutable factory creates visitors that construct logic expressions
 * from node IDs. It can be passed by const reference to \c build_logic.
 */
template<class LogicBuilder>
class StaticBuildLogicPolicy
{
    static_assert(std::is_invocable_v<LogicBuilder, NodeId>);

  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

  public:
    // Construct without mapping
    explicit StaticBuildLogicPolicy(CsgTree const& tree) : tree_{tree} {}
    // Construct with mapping
    StaticBuildLogicPolicy(CsgTree const& tree, VecSurface const& vs)
        : tree_{tree}, mapping_{&vs}
    {
    }

    //! Create a visitor for building logic
    auto operator()(VecLogic& logic) const
    {
        return LogicBuilder{tree_, logic, mapping_};
    }

  private:
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//

using PostfixBuildLogicPolicy = StaticBuildLogicPolicy<PostfixLogicBuilder>;
using InfixBuildLogicPolicy = StaticBuildLogicPolicy<InfixLogicBuilder>;

//---------------------------------------------------------------------------//
/*!
 * Runtime-dispatching policy for building logic expressions.
 *
 * This policy class selects between postfix and infix notation at runtime
 * based on the input \c LogicNotation enum value.
 */
class DynamicBuildLogicPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    using Builder = std::function<void(NodeId)>;
    //!@}

  public:
    // Construct with optional mapping
    DynamicBuildLogicPolicy(LogicNotation notation,
                            CsgTree const& tree,
                            VecSurface const* mapping);

    // Create a visitor for building logic
    Builder operator()(VecLogic& logic) const;

  private:
    LogicNotation notation_;
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
// TEMPLATE INSTANTIATION
//---------------------------------------------------------------------------//

extern template class BaseLogicBuilder<PostfixLogicBuilder>;
extern template class BaseLogicBuilder<InfixLogicBuilder>;

extern template BuildLogicResult
build_logic(PostfixBuildLogicPolicy const&, NodeId);
extern template BuildLogicResult
build_logic(InfixBuildLogicPolicy const&, NodeId);
extern template BuildLogicResult
build_logic(DynamicBuildLogicPolicy const&, NodeId);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
