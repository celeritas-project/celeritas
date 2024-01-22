//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/BoundingBoxUtils.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate bounding box enclosing bounding boxes for specified indices.
 */
inline FastBBox calc_union(std::vector<FastBBox> const& bboxes,
                           std::vector<LocalVolumeId> const& indices)
{
    FastBBox result;
    for (auto const& id : indices)
    {
        CELER_ASSERT(id < bboxes.size());
        result = calc_union(result, bboxes[id.unchecked_get()]);
    }

    return result;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
