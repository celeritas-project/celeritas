//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/ImageIO.cc
//---------------------------------------------------------------------------//
#include "ImageIO.hh"

#include "celeritas_cmake_strings.h"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "ImageStore.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
Real3 from_cm(Real3 const& r)
{
    using CmPoint = Quantity<units::Centimeter, Real3>;
    return native_value_from(CmPoint{r});
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, ImageRunArgs const& v)
{
    j = nlohmann::json{{"lower_left", v.lower_left},
                       {"upper_right", v.upper_right},
                       {"rightward_ax", v.rightward_ax},
                       {"vertical_pixels", v.vertical_pixels},
                       {"_units", celeritas_units}};
}

void from_json(nlohmann::json const& j, ImageRunArgs& v)
{
    v.lower_left = from_cm(j.at("lower_left").get<Real3>());
    v.upper_right = from_cm(j.at("upper_right").get<Real3>());
    j.at("rightward_ax").get_to(v.rightward_ax);
    j.at("vertical_pixels").get_to(v.vertical_pixels);
}

void to_json(nlohmann::json& j, ImageStore const& v)
{
    j = nlohmann::json{{"origin", v.origin()},
                       {"down_ax", v.down_ax()},
                       {"right_ax", v.right_ax()},
                       {"pixel_width", v.pixel_width()},
                       {"dims", v.dims()},
                       {"int_size", sizeof(int)},
                       {"_units", celeritas_units}};
}

//!@}
//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
