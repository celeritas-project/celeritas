//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>

#include "corecel/cont/Range.hh"
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

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
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

    constexpr auto max_real = std::numeric_limits<T>::max();

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        if (bbox.lower()[ax] > -max_real || bbox.upper()[ax] < max_real)
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the center of a bounding box.
 */
template<class T>
inline Array<T, 3> center(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> center;
    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        center[ax] = (bbox.lower()[ax] + bbox.upper()[ax]) / 2;
    }

    return center;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the surface area of a bounding box.
 */
template<class T>
inline T surface_area(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> lengths;

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
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
bbox_union(BoundingBox<T> const& a, BoundingBox<T> const& b)
{
    CELER_EXPECT(a && b);

    Array<T, 3> lower;
    Array<T, 3> upper;

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        lower[ax] = std::min(a.lower()[ax], b.lower()[ax]);
        upper[ax] = std::max(a.upper()[ax], b.upper()[ax]);
    }

    return BoundingBox<T>{lower, upper};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
