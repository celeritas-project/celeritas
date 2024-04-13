//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ImageInput;
class ImageParams;

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, ImageInput const&);
void from_json(nlohmann::json const& j, ImageInput&);

void to_json(nlohmann::json& j, ImageParams const&);
//---------------------------------------------------------------------------//
}  // namespace celeritas
