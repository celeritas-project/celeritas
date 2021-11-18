//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageIO.cc
//---------------------------------------------------------------------------//
#include "ImageIO.hh"

#include "base/Array.json.hh"
#include "ImageStore.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, const ImageRunArgs& v)
{
    j = nlohmann::json{{"lower_left", v.lower_left},
                       {"upper_right", v.upper_right},
                       {"rightward_ax", v.rightward_ax},
                       {"vertical_pixels", v.vertical_pixels}};
}

void from_json(const nlohmann::json& j, ImageRunArgs& v)
{
    j.at("lower_left").get_to(v.lower_left);
    j.at("upper_right").get_to(v.upper_right);
    j.at("rightward_ax").get_to(v.rightward_ax);
    j.at("vertical_pixels").get_to(v.vertical_pixels);
}

void to_json(nlohmann::json& j, const ImageStore& v)
{
    j = nlohmann::json{{"origin", v.origin()},
                       {"down_ax", v.down_ax()},
                       {"right_ax", v.right_ax()},
                       {"pixel_width", v.pixel_width()},
                       {"dims", v.dims()},
                       {"int_size", sizeof(int)}};
}

//!@}
//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
