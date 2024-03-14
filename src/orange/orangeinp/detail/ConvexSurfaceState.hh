//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ConvexSurfaceState.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "orange/surf/FaceNamer.hh"
#include "orange/transform/VariantTransform.hh"

#include "BoundingZone.hh"
#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Local state for building a set of intersected surfaces.
 *
 * Note that the surface clippers have *pointers* to local and global bounding
 * zones. Those must exceed the lifetime of this state.
 */
struct ConvexSurfaceState
{
    //!@{
    //! \name Input state
    //! Local-to-global transform
    VariantTransform const* transform{nullptr};
    //! Name of the object being built
    std::string object_name;
    //! Generate a name from a surface (has internal state)
    FaceNamer make_face_name;
    //!@}

    //!@{
    //! \name Output state
    //! Local (to convex surface state) interior/exterior
    BoundingZone local_bzone = BoundingZone::from_infinite();
    //! Global (to unit) interior/exterior
    BoundingZone global_bzone = BoundingZone::from_infinite();
    //! Inserted CSG nodes
    std::vector<NodeId> nodes;
    //!@}

    //! True if the state is valid
    explicit operator bool() const
    {
        return transform && !object_name.empty();
    }
};

//---------------------------------------------------------------------------//
// Use the local and global bounding zones to create a better zone
BoundingZone calc_merged_bzone(ConvexSurfaceState const& css);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
