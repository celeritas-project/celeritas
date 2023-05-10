//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/cont/ArrayIO.json.hh"

#include "BoundingBox.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read a quantity from a JSON file.
 */
inline void from_json(nlohmann::json const& j, BoundingBox& bbox)
{
    auto arrays = j.at("bbox").get<Array<Real3, 2>>();
    bbox = {arrays[0], arrays[1]};
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
