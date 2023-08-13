//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Host/device functions
//---------------------------------------------------------------------------//
/*!
 * Determine if a point is contained in a bounding box.
 */
template<class T, class U>
inline CELER_FUNCTION bool
is_inside(BoundingBox<T> const& bbox, Array<U, 3> point)
{
    CELER_EXPECT(bbox);

    for (auto ax : range(to_int(Axis::size_)))
    {
        if (point[ax] < bbox.lower()[ax] || point[ax] > bbox.upper()[ax])
        {
            return false;
        }
    }

    return true;
}

//---------------------------------------------------------------------------//
// Host-only functions
//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box spans (-inf, inf) in every direction.
 */
template<class T>
inline bool is_infinite(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    return std::all_of(bbox.lower().begin(), bbox.lower().end(), std::isinf<T>)
           && std::all_of(
               bbox.upper().begin(), bbox.upper().end(), std::isinf<T>);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the center of a bounding box.
 */
template<class T>
inline Array<T, 3> calc_center(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> center;
    for (auto ax : range(to_int(Axis::size_)))
    {
        center[ax] = (bbox.lower()[ax] + bbox.upper()[ax]) / 2;
    }

    return center;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the surface area of a bounding box.
 */
template<class T>
inline T calc_surface_area(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> lengths;

    for (auto ax : range(to_int(Axis::size_)))
    {
        lengths[ax] = bbox.upper()[ax] - bbox.lower()[ax];
    }

    return 2
           * (lengths[to_int(Axis::x)] * lengths[to_int(Axis::y)]
              + lengths[to_int(Axis::x)] * lengths[to_int(Axis::z)]
              + lengths[to_int(Axis::y)] * lengths[to_int(Axis::z)]);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the smallest bounding box enclosing two bounding boxes.
 */
template<class T>
inline BoundingBox<T>
calc_union(BoundingBox<T> const& a, BoundingBox<T> const& b)
{
    Array<T, 3> lower;
    Array<T, 3> upper;

    for (auto ax : range(to_int(Axis::size_)))
    {
        lower[ax] = celeritas::min(a.lower()[ax], b.lower()[ax]);
        upper[ax] = celeritas::max(a.upper()[ax], b.upper()[ax]);
    }

    return BoundingBox<T>{lower, upper};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
