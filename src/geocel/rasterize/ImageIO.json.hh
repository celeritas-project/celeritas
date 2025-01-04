//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
