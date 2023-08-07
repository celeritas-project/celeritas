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
template<class T>
inline void fix_inf(typename celeritas::BoundingBox<T>::Real3* point)
{
    constexpr auto max_real = std::numeric_limits<
        typename celeritas::BoundingBox<T>::value_type>::max();
    constexpr auto inf = std::numeric_limits<
        typename celeritas::BoundingBox<T>::value_type>::infinity();

    for (auto axis : range(celeritas::Axis::size_))
    {
        auto ax = to_int(axis);
        if (std::fabs((*point)[ax]) == max_real)
        {
            (*point)[ax] = std::copysign(inf, (*point)[ax]);
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
template<class T>
inline void from_json(nlohmann::json const& j, BoundingBox<T>& bbox)
{
    CELER_VALIDATE(j.size() == 2,
                   << " bounding box must have lower and upper extents");

    using Real3 = typename BoundingBox<T>::Real3;
    auto lower = j[0].get<Real3>();
    auto upper = j[1].get<Real3>();

    fix_inf<T>(&lower);
    fix_inf<T>(&upper);

    bbox = BoundingBox<T>{lower, upper};
}

//---------------------------------------------------------------------------//
/*!
 * Write an array to a JSON file.
 */
template<class T>
inline void to_json(nlohmann::json& j, BoundingBox<T> const& bbox)
{
    j = {bbox.lower(), bbox.upper()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
