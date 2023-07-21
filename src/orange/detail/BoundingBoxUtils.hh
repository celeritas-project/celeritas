//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BoundingBoxUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>

#include "corecel/cont/Range.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box spans (-inf, inf) in every direction.
 */
inline CELER_FUNCTION bool is_infinite(BoundingBox const& bbox)
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
 * Create a vector of axes sorted from longest to shortest.
 */
inline CELER_FUNCTION std::vector<Axis> sort_axes(BoundingBox const& bbox)
{
    CELER_EXPECT(bbox);

    std::vector<Axis> axes;
    std::vector<real_type> lengths;

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        axes.push_back(axis);
        lengths.push_back(bbox.upper()[ax] - bbox.lower()[ax]);
    }

    std::sort(axes.begin(), axes.end(), [&](Axis axis1, Axis axis2) {
        return lengths[to_int(axis1)] > lengths[to_int(axis2)];
    });
    return axes;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the centers of each bounding box.
 */
inline CELER_FUNCTION Real3 center(BoundingBox const& bbox)
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
 * Calculate bounding box enclosing two bounding boxes.
 */
inline CELER_FUNCTION BoundingBox bbox_union(BoundingBox const& a,
                                             BoundingBox const& b)
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
inline CELER_FUNCTION BoundingBox
bbox_union(std::vector<BoundingBox> const& bboxes,
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
}  // namespace detail
}  // namespace celeritas
