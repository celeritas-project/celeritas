//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include "base/Macros.hh"
#include "Types.hh"
#include "../VolumeView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// FUNCTION-LIKE CLASSES
//---------------------------------------------------------------------------//
/*!
 * Predicate for finding closest valid distance.
 */
struct CloserNonzeroDistance
{
    CELER_CONSTEXPR_FUNCTION bool operator()(real_type a, real_type b) const
    {
        return a > 0 && (b <= 0 || a < b);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Predicate for partitioning valid (finite nonzero) from invalid distances.
 */
struct IntersectionPartitioner
{
    const TempNextFace& temp_next;

    CELER_FUNCTION bool operator()(size_type isect) const
    {
        CELER_ASSERT(isect < temp_next.size);
        const real_type distance = temp_next.distance[isect];
        return distance > 0 && !std::isinf(distance);
    }
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
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
