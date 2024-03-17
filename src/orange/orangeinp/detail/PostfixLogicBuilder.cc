//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/PostfixLogicBuilder.cc
//---------------------------------------------------------------------------//
#include "PostfixLogicBuilder.hh"

#include "corecel/cont/VariantUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with postfix operation.
 *
 * This is an implementation detail of the \c PostfixLogicBuilder class. The
 * user invokes this class with a node ID (usually representing a cell), and
 * then this class recurses into the daughters using a tree visitor.
 */
class PostfixLogicBuilderImpl
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
    inline PostfixLogicBuilderImpl(CsgTree const& tree,
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

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_;
    VecLogic* logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to the logic expression.
 *
 * The surface mapping vector is *optional*.
 */
PostfixLogicBuilderImpl::PostfixLogicBuilderImpl(CsgTree const& tree,
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
void PostfixLogicBuilderImpl::operator()(NodeId const& n)
{
    visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
void PostfixLogicBuilderImpl::operator()(True const&)
{
    logic_->push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
void PostfixLogicBuilderImpl::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
void PostfixLogicBuilderImpl::operator()(Surface const& s)
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
 * TODO: aliased node shouldn't be reachable if we're fully simplified.
 */
void PostfixLogicBuilderImpl::operator()(Aliased const& n)
{
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit a negated node and append 'not'.
 */
void PostfixLogicBuilderImpl::operator()(Negated const& n)
{
    (*this)(n.node);
    logic_->push_back(logic::lnot);
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes and append the conjunction.
 */
void PostfixLogicBuilderImpl::operator()(Joined const& n)
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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Convert a single node to postfix notation.
 *
 * The per-node local surfaces (faces) are sorted in ascending order of ID, not
 * of access, since they're always evaluated sequentially rather than as part
 * of the logic evaluation itself.
 */
auto PostfixLogicBuilder::operator()(NodeId n) const -> result_type
{
    CELER_EXPECT(n < tree_.size());

    // Construct logic vector as local surface IDs
    VecLogic lgc;
    PostfixLogicBuilderImpl build_impl{tree_, mapping_, &lgc};
    build_impl(n);

    // Construct sorted vector of faces
    std::vector<LocalSurfaceId> faces;
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

    return {std::move(faces), std::move(lgc)};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
