//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Macros.hh"
#include "base/NumericLimits.hh"

#include "../VolumeView.hh"
#include "Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// FUNCTION-LIKE CLASSES
//---------------------------------------------------------------------------//
/*!
 * Predicate for partitioning valid (finite positive) from invalid distances.
 */
struct IsIntersectionFinite
{
    const TempNextFace& temp_next;

    CELER_FUNCTION bool operator()(size_type isect) const
    {
        CELER_ASSERT(isect < temp_next.size);
        const real_type distance = temp_next.distance[isect];
        return distance < numeric_limits<real_type>::max();
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Convert an OnSurface (may be null) to an OnFace using a volume view.
 */
inline CELER_FUNCTION OnFace find_face(const VolumeView& vol, OnSurface surf)
{
    return {surf ? vol.find_face(surf.id()) : FaceId{}, surf.unchecked_sense()};
}

//---------------------------------------------------------------------------//
/*!
 * Convert an OnFace (may be null) to an OnSurface using a volume view.
 */
inline CELER_FUNCTION OnSurface get_surface(const VolumeView& vol, OnFace face)
{
    return {face ? vol.get_surface(face.id()) : SurfaceId{},
            face.unchecked_sense()};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
