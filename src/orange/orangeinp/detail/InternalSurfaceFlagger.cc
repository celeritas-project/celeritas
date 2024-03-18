//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/InternalSurfaceFlagger.cc
//---------------------------------------------------------------------------//
#include "InternalSurfaceFlagger.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a tree.
 */
InternalSurfaceFlagger::InternalSurfaceFlagger(CsgTree const& tree)
    : tree_{tree}, cache_(tree.size(), Status::unknown)
{
}

//---------------------------------------------------------------------------//
/*!
 * Visit a node, using the cache, to determine its flag.
 */
bool InternalSurfaceFlagger::operator()(NodeId const& n)
{
    CELER_EXPECT(n < cache_.size());
    Status& status = cache_[n.get()];
    if (status == Status::unknown)
    {
        Node const& node = tree_[n];
        CELER_ASSUME(!node.valueless_by_exception());
        status = static_cast<Status>(std::visit(*this, node));
    }
    CELER_ENSURE(status >= 0);
    return status;
}

//---------------------------------------------------------------------------//
/*!
 * Aliased nodes forward to the alias.
 */
bool InternalSurfaceFlagger::operator()(Aliased const& n)
{
    return (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit a negated node.
 *
 * Negated "and" nodes are *not* simple. Negated "or" nodes might be, but we
 * ignore that for now.
 */
bool InternalSurfaceFlagger::operator()(Negated const& n)
{
    if (auto* j = std::get_if<Joined>(&tree_[n.node]))
    {
        // Pointee is a "joined" node
        return Status::internal;
    }

    // Otherwise forward on the result (for simplified trees, this should just
    // be a surface or "true")
    return (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Intersections of "simple" nodes are simple; unions are never.
 */
bool InternalSurfaceFlagger::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);

    if (n.op == op_or)
    {
        return Status::internal;
    }

    for (NodeId const& d : n.nodes)
    {
        if ((*this)(d) == Status::internal)
        {
            return Status::internal;
        }
    }

    return Status::simple;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
