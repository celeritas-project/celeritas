//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/InternalSurfaceFlagger.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/VariantUtils.hh"

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
 * Return whether a volume has an internal surface crossing.
 *
 * In a "simple" volume with no internal surface crossings, any intersection
 * with any surface guarantees that a track will exit the volume at that
 * boundary at that distance.
 *
 * \return True if internal surface crossings may be present
 */
class InternalSurfaceFlagger
{
  public:
    // Construct from a tree
    explicit InternalSurfaceFlagger(CsgTree const& tree);

    //!@{
    //! \name Visit a node directly

    // No surface crossings
    bool operator()(True const&) { return simple; }
    // False is never explicitly part of the node tree
    bool operator()(False const&) { CELER_ASSERT_UNREACHABLE(); }
    // Surfaces are 'simple'
    bool operator()(Surface const&) { return simple; }
    // Aliased nodes forward to the alias
    bool operator()(Aliased const&);
    // Negated nodes may have internal crossings if they negate "joined"
    bool operator()(Negated const&);
    // Intersections of "simple" nodes are simple; unions are never
    bool operator()(Joined const&);
    //!@}

    // Visit a node, using the cache, to determine its flag
    bool operator()(NodeId const& n);

  private:
    //// TYPES ////

    enum Status : signed char
    {
        unknown = -1,  //!< Node has not yet been evaluated
        simple = 0,  //!< Known not to have reentrant surfaces
        internal = 1  //!< Known to have reentrant surfaces
    };

    //// DATA ////

    CsgTree const& tree_;
    std::vector<Status> cache_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
