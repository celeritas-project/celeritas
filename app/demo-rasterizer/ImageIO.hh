//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/ImageIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace app
{
class ImageStore;
//---------------------------------------------------------------------------//
//! Image construction arguments
struct ImageRunArgs
{
    Real3 lower_left;
    Real3 upper_right;
    Real3 rightward_ax;
    unsigned int vertical_pixels;
};

void to_json(nlohmann::json& j, ImageRunArgs const& value);
void from_json(nlohmann::json const& j, ImageRunArgs& value);

void to_json(nlohmann::json& j, ImageStore const& value);
//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
