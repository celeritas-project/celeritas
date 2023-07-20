//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BoundingBoxUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/BoundingBox.hh"

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
}  // namespace detail
}  // namespace celeritas
