//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildLogic.cc
//---------------------------------------------------------------------------//
#include "BuildLogic.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

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
 * Sort the faces of a volume and remap the logic expression.
 */
BuildLogicResult::VecSurface remap_faces(BuildLogicResult::VecLogic& lgc)
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

}  // namespace

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
    CELER_EXPECT(logic_.empty());
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
    this->push_back(logic::ltrue);
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

    this->push_back(sidx);
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
 * Append negation after a node.
 */
void PostfixLogicBuilder::operator()(Negated const& n)
{
    (*this)(n.node);
    this->push_back(logic::lnot);
}

//---------------------------------------------------------------------------//
/*!
 * Push multiply joined nodes.
 */
void PostfixLogicBuilder::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);

    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        (*this)(*iter++);
        this->push_back(n.op);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Negate an expression.
 */
void InfixLogicBuilder::operator()(Negated const& n)
{
    this->push_back(logic::lnot);
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Join a set of nodes.
 *
 * This omits the outer parentheses.
 */
void InfixLogicBuilder::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);
    if (depth_ > 0)
    {
        this->push_back(logic::lopen);
    }
    ++depth_;
    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        this->push_back(n.op);
        (*this)(*iter++);
    }
    --depth_;
    if (depth_ > 0)
    {
        this->push_back(logic::lclose);
    }
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
template<class LogicPolicy>
BuildLogicResult build_logic(LogicPolicy const& policy, NodeId n)
{
    // static_assert(std::is_invocable_v<LogicPolicy,
    // BuildLogicResult::VecLogic>);

    // Construct logic vector as local surface IDs
    BuildLogicResult::VecLogic lgc;
    auto build_impl = policy(lgc);
    build_impl(n);

    return {remap_faces(lgc), std::move(lgc)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optional mapping.
 */
DynamicBuildLogicPolicy::DynamicBuildLogicPolicy(LogicNotation notation,
                                                 CsgTree const& tree,
                                                 VecSurface const* mapping)
    : notation_{notation}, tree_{tree}, mapping_{mapping}
{
}

//---------------------------------------------------------------------------//
/*!
 * Create a visitor for building logic.
 */
auto DynamicBuildLogicPolicy::operator()(VecLogic& logic) const -> Builder
{
    CELER_EXPECT(logic.empty());

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

//---------------------------------------------------------------------------//
// TEMPLATE INSTANTIATION
//---------------------------------------------------------------------------//

template class BaseLogicBuilder<PostfixLogicBuilder>;
template class BaseLogicBuilder<InfixLogicBuilder>;

template BuildLogicResult build_logic(PostfixBuildLogicPolicy const&, NodeId);
template BuildLogicResult build_logic(InfixBuildLogicPolicy const&, NodeId);
template BuildLogicResult build_logic(DynamicBuildLogicPolicy const&, NodeId);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
