//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxIO.json.cc
//---------------------------------------------------------------------------//
#include "BoundingBoxIO.json.hh"

#include <cmath>
#include <limits>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Replace "max" with "inf" since the latter can't be represented in JSON.
template<class T>
void max_to_inf(celeritas::Array<T, 3>* point)
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
//! Replace "inf" with "max" since the former can't be represented in JSON.
template<class T>
void inf_to_max(celeritas::Array<T, 3>* point)
{
    constexpr auto max_real = std::numeric_limits<T>::max();

    for (auto axis : range(celeritas::Axis::size_))
    {
        auto ax = to_int(axis);
        if (std::isinf((*point)[ax]))
        {
            (*point)[ax] = std::copysign(max_real, (*point)[ax]);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Read a bounding box from a JSON file.
 *
 * A bounding box is a pair of lower/upper points, with the exception of a
 * "null" (enclosing no points) bounding box.
 */
template<class T>
void from_json(nlohmann::json const& j, BoundingBox<T>& bbox)
{
    if (j.is_null())
    {
        // Special case: null bounding box
        bbox = {};
        return;
    }

    CELER_VALIDATE(j.is_array() && j.size() == 2,
                   << "bounding box must have lower and upper extents");

    // Replace large values (substituted from +-inf) with inf
    auto lower = j[0].get<Array<T, 3>>();
    auto upper = j[1].get<Array<T, 3>>();
    max_to_inf(&lower);
    max_to_inf(&upper);

    bbox = BoundingBox<T>{lower, upper};
}

//---------------------------------------------------------------------------//
/*!
 * Write a bounding box to a JSON file.
 */
template<class T>
void to_json(nlohmann::json& j, BoundingBox<T> const& bbox)
{
    if (!bbox)
    {
        // Special case: null bounding box
        j = nullptr;
        return;
    }

    // Replace unrepresentable infinities with large values
    auto lower = bbox.lower();
    auto upper = bbox.upper();
    inf_to_max(&lower);
    inf_to_max(&upper);

    j = nlohmann::json::array({lower, upper});
}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATION
//---------------------------------------------------------------------------//

template void from_json(nlohmann::json const&, BoundingBox<float>&);
template void to_json(nlohmann::json&, BoundingBox<float> const&);
template void from_json(nlohmann::json const&, BoundingBox<double>&);
template void to_json(nlohmann::json&, BoundingBox<double> const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
