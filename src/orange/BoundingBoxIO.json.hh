//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

#include "corecel/cont/ArrayIO.json.hh"

#include "BoundingBox.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Replace "max" with "inf" since the latter can't be represented in JSON.
template<class T>
inline void fix_inf(celeritas::Array<T, 3>* point)
{
    constexpr auto max_real = std::numeric_limits<T>::max();
    constexpr auto inf = std::numeric_limits<T>::infinity();

    for (auto axis : range(celeritas::Axis::size_))
    {
        auto ax = to_int(axis);
        if (std::fabs((*point)[ax]) == max_real)
        {
            (*point)[ax] = std::copysign(inf, (*point)[ax]);
        }
    }
}
//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Read a bounding box from a JSON file.
 *
 * A bounding box can be either \c null , indicating an infinite or unknown
 * bounding box, or a pair of lower/upper points.
 */
template<class T>
inline void from_json(nlohmann::json const& j, BoundingBox<T>& bbox)
{
    if (j.is_null())
    {
        // Missing bounding box
        bbox = BoundingBox<T>::from_infinite();
        return;
    }

    CELER_VALIDATE(j.is_array() && j.size() == 2,
                   << "bounding box must have lower and upper extents");

    auto lower = j[0].get<Array<T, 3>>();
    auto upper = j[1].get<Array<T, 3>>();

    detail::fix_inf(&lower);
    detail::fix_inf(&upper);

    bbox = BoundingBox<T>{lower, upper};
}

//---------------------------------------------------------------------------//
/*!
 * Write a bounding box to a JSON file.
 */
template<class T>
inline void to_json(nlohmann::json& j, BoundingBox<T> const& bbox)
{
    j = {bbox.lower(), bbox.upper()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
