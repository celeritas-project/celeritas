//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <nlohmann/json.hpp>

#include "corecel/cont/ArrayIO.json.hh"

#include "BoundingBox.hh"

namespace
{
void fix_inf(celeritas::Real3* point)
{
    static constexpr auto max_real
        = std::numeric_limits<celeritas::real_type>::max();
    static constexpr auto inf
        = std::numeric_limits<celeritas::real_type>::infinity();

    for (auto axis : range(celeritas::Axis::size_))
    {
        auto ax = to_int(axis);
        if ((*point)[ax] == max_real)
        {
            (*point)[ax] = inf;
        }
    }
}
}  // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read a quantity from a JSON file.
 */
inline void from_json(nlohmann::json const& j, BoundingBox& bbox)
{
    CELER_VALIDATE(j.size() == 2,
                   << " bounding box must have lower and upper extents");

    auto lower = j[0].get<Real3>();
    auto upper = j[1].get<Real3>();

    fix_inf(&lower);
    fix_inf(&upper);

    bbox = {lower, upper};
}

//---------------------------------------------------------------------------//
/*!
 * Write an array to a JSON file.
 */
inline void to_json(nlohmann::json& j, BoundingBox const& bbox)
{
    j = {bbox.lower(), bbox.upper()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
