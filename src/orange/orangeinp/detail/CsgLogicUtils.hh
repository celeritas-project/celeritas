//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgLogicUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>
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
template<class BuildLogicPolicy>
inline BuildLogicResult build_logic(BuildLogicPolicy const& policy, NodeId n)
{
    // Construct logic vector as local surface IDs
    BuildLogicResult::VecLogic lgc;
    auto visitor = policy(lgc);
    visitor(n);
    return {remap_faces(lgc), std::move(lgc)};
}

//---------------------------------------------------------------------------//
/*!
 * Base class for logic builder visitors following CRTP pattern.
 *
 * Visitors recursively traverse the CSG tree and append to a logic vector.
 * The call operator for Negated and Joined are not implemented in the base
 * visitor and must be provided by the derived class.
 */
template<class BuilderVisitor>
class BaseBuildLogicVisitor
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
    // Construct without mapping
    inline BaseBuildLogicVisitor(CsgTree const& tree, VecLogic& logic);
    // Construct with optional mapping
    inline BaseBuildLogicVisitor(CsgTree const& tree,
                                 VecLogic& logic,
                                 VecSurface const& vs);

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
    //! Access the logic expression directly
    VecLogic& logic() { return logic_; }

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_{nullptr};
    VecLogic& logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct without mapping.
 */
template<class BuilderVisitor>
BaseBuildLogicVisitor<BuilderVisitor>::BaseBuildLogicVisitor(CsgTree const& tree,
                                                             VecLogic& logic)
    : visit_node_{tree}, logic_{logic}
{
    static_assert(std::is_base_of_v<BaseBuildLogicVisitor, BuilderVisitor>,
                  "CRTP: template parameter must be derived class");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optional mapping.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 */
template<class BuilderVisitor>
BaseBuildLogicVisitor<BuilderVisitor>::BaseBuildLogicVisitor(
    CsgTree const& tree, VecLogic& logic, VecSurface const& vs)
    : visit_node_{tree}, mapping_{&vs}, logic_{logic}
{
    static_assert(std::is_base_of_v<BaseBuildLogicVisitor, BuilderVisitor>,
                  "CRTP: template parameter must be derived class");
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
template<class BuilderVisitor>
void BaseBuildLogicVisitor<BuilderVisitor>::operator()(NodeId const& n)
{
    visit_node_(static_cast<BuilderVisitor&>(*this), n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
template<class BuilderVisitor>
void BaseBuildLogicVisitor<BuilderVisitor>::operator()(True const&)
{
    logic_.push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
template<class BuilderVisitor>
void BaseBuildLogicVisitor<BuilderVisitor>::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
template<class BuilderVisitor>
void BaseBuildLogicVisitor<BuilderVisitor>::operator()(Surface const& s)
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
template<class BuilderVisitor>
void BaseBuildLogicVisitor<BuilderVisitor>::operator()(Aliased const& n)
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
class PostfixBuildLogicVisitor
    : public BaseBuildLogicVisitor<PostfixBuildLogicVisitor>
{
  public:
    using BaseBuildLogicVisitor::BaseBuildLogicVisitor;

    //!@{
    //! \name Visit a node directly
    using BaseBuildLogicVisitor::operator();

    //! Visit a negated node and append 'not'.
    void operator()(Negated const& n)
    {
        (*this)(n.node);
        logic().push_back(logic::lnot);
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
            logic().push_back(n.op);
        }
    }
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Policy for building logic in postfix notation.
 *
 * This immutable factory creates visitors that construct logic expressions
 * in postfix notation. It can be passed by const reference to \c build_logic.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "0 1 & 2 & &"}
    all(1, 3, !all(2, 4)) -> {{1, 2, 3, 4}, "0 2 & 1 3 & ~ &"}
 * \endverbatim
 */
class PostfixBuildLogicPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

  public:
    // Construct without mapping
    explicit PostfixBuildLogicPolicy(CsgTree const& tree) : tree_{tree} {}
    // Construct with optional mapping
    PostfixBuildLogicPolicy(CsgTree const& tree, VecSurface const& vs)
        : tree_{tree}, mapping_{&vs}
    {
    }

    //! Create a visitor for building logic
    auto operator()(VecLogic& logic) const
    {
        if (mapping_)
        {
            return PostfixBuildLogicVisitor{tree_, logic, *mapping_};
        }
        return PostfixBuildLogicVisitor{tree_, logic};
    }

  private:
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
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
class InfixBuildLogicVisitor
    : public BaseBuildLogicVisitor<InfixBuildLogicVisitor>
{
  public:
    using BaseBuildLogicVisitor::BaseBuildLogicVisitor;

    //!@{
    //! \name Visit a node directly
    using BaseBuildLogicVisitor::operator();

    //! Append 'not' and visit a negated node.
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
/*!
 * Policy for building logic in infix notation.
 *
 * This immutable factory creates visitors that construct logic expressions
 * in infix notation. It can be passed by const reference to \c build_logic.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "(0 & 1 & 2)"}
    all(1, 3, any(~(2), ~(4))) -> {{1, 2, 3, 4}, "(0 & 2 & (~1 | ~3))"}
 * \endverbatim
 */
class InfixBuildLogicPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

  public:
    // Construct without mapping
    explicit InfixBuildLogicPolicy(CsgTree const& tree) : tree_{tree} {}
    // Construct with optional mapping
    InfixBuildLogicPolicy(CsgTree const& tree, VecSurface const& vs)
        : tree_{tree}, mapping_{&vs}
    {
    }

    //! Create a visitor for building logic
    auto operator()(VecLogic& logic) const
    {
        if (mapping_)
        {
            return InfixBuildLogicVisitor{tree_, logic, *mapping_};
        }
        return InfixBuildLogicVisitor{tree_, logic};
    }

  private:
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
