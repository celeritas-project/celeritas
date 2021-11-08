//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceInputIO.json.cc
//---------------------------------------------------------------------------//
#include "SurfaceInputIO.json.hh"

#include <algorithm>
#include <vector>
#include "base/Range.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Build a vector of strings for each surface type.
 */
std::vector<std::string> make_surface_strings()
{
    std::vector<std::string> result(static_cast<size_type>(SurfaceType::size_));
    for (auto surf_type : range(SurfaceType::size_))
    {
        result[static_cast<size_type>(surf_type)] = to_cstring(surf_type);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a surface type string to an enum for I/O.
 */
SurfaceType to_surface_type(const std::string& s)
{
    // The number of surface types will be short, and presumably each string is
    // small enough to fit inside string's static allocation. Therefore the
    // string search will be on a small-ish, nearly contiguous block of memory,
    // so it's preferable than using unordered_map or a more heavyweight
    // container.
    static const auto surface_string = make_surface_strings();

    auto iter = std::find(surface_string.begin(), surface_string.end(), s);
    CELER_VALIDATE(iter != surface_string.end(),
                   << "invalid surface string '" << s << "'");

    unsigned int result_int = iter - surface_string.begin();
    CELER_EXPECT(result_int < static_cast<size_type>(SurfaceType::size_));
    return static_cast<SurfaceType>(result_int);
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * I/O routine for JSON.
 */
void from_json(const nlohmann::json& j, SurfaceInput& value)
{
    // Read and convert types
    auto type_labels = j.at("types").get<std::vector<std::string>>();
    value.types.resize(type_labels.size());
    std::transform(type_labels.begin(),
                   type_labels.end(),
                   value.types.begin(),
                   &to_surface_type);

    j.at("data").get_to(value.data);
    j.at("sizes").get_to(value.sizes);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
