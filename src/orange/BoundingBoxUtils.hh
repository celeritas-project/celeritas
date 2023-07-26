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
inline CELER_FUNCTION bool is_inside(BoundingBox const& bbox, Real3 point)
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
inline bool is_infinite(BoundingBox const& bbox)
{
    CELER_EXPECT(bbox);

    auto max_real = std::numeric_limits<real_type>::max();

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
inline Real3 center(BoundingBox const& bbox)
{
    CELER_EXPECT(bbox);

    Real3 center;
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
inline real_type surface_area(BoundingBox const& bbox)
{
    CELER_EXPECT(bbox);

    Array<real_type, to_int(Axis::size_)> lengths;

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
 * Calculate bounding box enclosing two bounding boxes.
 */
inline BoundingBox bbox_union(BoundingBox const& a, BoundingBox const& b)
{
    CELER_EXPECT(a && b);

    Real3 lower, upper;

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        lower[ax] = std::min(a.lower()[ax], b.lower()[ax]);
        upper[ax] = std::max(a.upper()[ax], b.upper()[ax]);
    }

    return {lower, upper};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate bounding box enclosing bounding boxes for specified indices.
 */
inline BoundingBox bbox_union(std::vector<BoundingBox> const& bboxes,
                              std::vector<LocalVolumeId> const& indices)
{
    CELER_EXPECT(!bboxes.empty());
    CELER_EXPECT(!indices.empty());

    auto result = bboxes[indices.front().unchecked_get()];

    for (auto id = std::next(indices.begin()); id != indices.end(); ++id)
    {
        result = bbox_union(result, bboxes[id->unchecked_get()]);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
