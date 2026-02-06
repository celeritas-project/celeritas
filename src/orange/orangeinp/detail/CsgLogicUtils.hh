//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgLogicUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>
#include <variant>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

#include "../CsgTree.hh"
#include "../CsgTypes.hh"

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
 * Sort the faces of a volume and remap the logic expression.
 */
inline BuildLogicResult::VecSurface remap_faces(BuildLogicResult::VecLogic& lgc)
{
    // Construct sorted vector of faces
    BuildLogicResult::VecSurface faces;
    for (auto const& v : lgc)
    {
        if (!logic::is_operator_token(v))
        {
            faces.push_back(LocalSurfaceId{v});
        }
    }

    // Sort and uniquify the vector
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());

    // Remap logic
    for (auto& v : lgc)
    {
        if (!logic::is_operator_token(v))
        {
            auto iter
                = find_sorted(faces.begin(), faces.end(), LocalSurfaceId{v});
            CELER_ASSUME(iter != faces.end());
            v = iter - faces.begin();
        }
    }
    return faces;
}

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
    inline BaseLogicBuilder(CsgTree const& tree,
                            VecLogic& logic,
                            VecSurface const* vs = nullptr);
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

  protected:
    VecLogic& logic_;

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * Construct with optional mapping pointer.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 */
template<class Impl>
BaseLogicBuilder<Impl>::BaseLogicBuilder(CsgTree const& tree,
                                         VecLogic& logic,
                                         VecSurface const* vs)
    : logic_{logic}, visit_node_{tree}, mapping_{vs}
{
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
template<class Impl>
void BaseLogicBuilder<Impl>::operator()(NodeId const& n)
{
    visit_node_(static_cast<Impl&>(*this), n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
template<class Impl>
void BaseLogicBuilder<Impl>::operator()(True const&)
{
    logic_.push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
template<class Impl>
void BaseLogicBuilder<Impl>::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
template<class Impl>
void BaseLogicBuilder<Impl>::operator()(Surface const& s)
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
template<class Impl>
void BaseLogicBuilder<Impl>::operator()(Aliased const& n)
{
    (*this)(n.node);
}

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

    //! Visit a negated node and append 'not'.
    void operator()(Negated const& n)
    {
        (*this)(n.node);
        logic_.push_back(logic::lnot);
    }

    //! Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n)
    {
        CELER_EXPECT(n.nodes.size() > 1);

        // Visit first node, then add conjunction for subsequent nodes
        auto iter = n.nodes.begin();
        (*this)(*iter++);

        while (iter != n.nodes.end())
        {
            (*this)(*iter++);
            logic_.push_back(n.op);
        }
    }
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Visitor for constructing logic in infix notation.
 *
 * Example: \verbatim
    all(1, 3, 5) -> "(0 & 1 & 2)"
    all(1, 3, any(~(2), ~(4))) -> "(0 & 2 & (~1 | ~3))"
 * \endverbatim
 */
class InfixLogicBuilder : public BaseLogicBuilder<InfixLogicBuilder>
{
  public:
    using BaseLogicBuilder::BaseLogicBuilder;

    //!@{
    //! \name Visit a node directly
    using BaseLogicBuilder::operator();

    //! Append 'not' and visit a negated node.
    void operator()(Negated const& n)
    {
        logic_.push_back(logic::lnot);
        (*this)(n.node);
    }

    //! Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n)
    {
        CELER_EXPECT(n.nodes.size() > 1);
        auto& logic = logic_;
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
inline BuildLogicResult build_logic(LogicPolicy const& policy, NodeId n)
{
    // Construct logic vector as local surface IDs
    BuildLogicResult::VecLogic lgc;
    auto visitor = policy(lgc);

    // Handle both direct builders and variant-wrapped builders
    if constexpr (std::is_same_v<
                      decltype(visitor),
                      std::variant<PostfixLogicBuilder, InfixLogicBuilder>>)
    {
        std::visit([n](auto& v) { v(n); }, visitor);
    }
    else
    {
        visitor(n);
    }

    return {remap_faces(lgc), std::move(lgc)};
}

//---------------------------------------------------------------------------//
/*!
 * Policy for building logic expressions.
 *
 * This immutable factory creates visitors that construct logic expressions.
 * It can be passed by const reference to \c build_logic.
 *
 * \tparam LogicBuilder The builder type (PostfixLogicBuilder or
 * InfixLogicBuilder)
 */
template<class LogicBuilder>
class BuildLogicPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

  public:
    // Construct without mapping
    explicit BuildLogicPolicy(CsgTree const& tree) : tree_{tree} {}
    // Construct with mapping
    BuildLogicPolicy(CsgTree const& tree, VecSurface const& vs)
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
/*!
 * Policy classes are factories.
 */
using PostfixBuildLogicPolicy = BuildLogicPolicy<PostfixLogicBuilder>;
using InfixBuildLogicPolicy = BuildLogicPolicy<InfixLogicBuilder>;

//---------------------------------------------------------------------------//
/*!
 * Runtime-dispatching policy for building logic expressions.
 *
 * This policy class selects between postfix and infix notation at runtime
 * based on a LogicNotation enum value. The operator() returns a variant
 * containing the appropriate builder type.
 */
class RuntimeBuildLogicPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    using Builder = std::variant<PostfixLogicBuilder, InfixLogicBuilder>;
    //!@}

  public:
    // Construct without mapping
    RuntimeBuildLogicPolicy(LogicNotation notation, CsgTree const& tree)
        : notation_{notation}, tree_{tree}
    {
    }
    // Construct with mapping
    RuntimeBuildLogicPolicy(LogicNotation notation,
                            CsgTree const& tree,
                            VecSurface const& vs)
        : notation_{notation}, tree_{tree}, mapping_{&vs}
    {
    }

    //! Create a visitor for building logic
    Builder operator()(VecLogic& logic) const
    {
        switch (notation_)
        {
            case LogicNotation::postfix:
                return PostfixLogicBuilder{tree_, logic, mapping_};
            case LogicNotation::infix:
                return InfixLogicBuilder{tree_, logic, mapping_};
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }

  private:
    LogicNotation notation_;
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
