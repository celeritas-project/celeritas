//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BIHUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/BoundingBoxUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate bounding box enclosing bounding boxes for specified indices.
 */
inline FastBBox bbox_union(std::vector<FastBBox> const& bboxes,
                           std::vector<LocalVolumeId> const& indices)
{
    CELER_EXPECT(!bboxes.empty());
    CELER_EXPECT(!indices.empty());

    auto id = indices.begin();
    auto result = bboxes[id->unchecked_get()];
    ++id;
    for (; id != indices.end(); ++id)
    {
        result = bbox_union(result, bboxes[id->unchecked_get()]);
    }

    return result;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
