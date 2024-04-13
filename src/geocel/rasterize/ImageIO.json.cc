//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageIO.json.cc
//---------------------------------------------------------------------------//
#include "ImageIO.json.hh"

#include "celeritas_cmake_strings.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "Image.hh"

namespace celeritas
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
void to_json(nlohmann::json& j, ImageInput const& v)
{
    j = nlohmann::json{{"lower_left", v.lower_left},
                       {"upper_right", v.upper_right},
                       {"rightward", v.rightward},
                       {"vertical_pixels", v.vertical_pixels},
                       {"_units", celeritas_units}};
}

void from_json(nlohmann::json const& j, ImageInput& v)
{
    v.lower_left = from_cm(j.at("lower_left").get<Real3>());
    v.upper_right = from_cm(j.at("upper_right").get<Real3>());
    j.at("rightward").get_to(v.rightward);
    j.at("vertical_pixels").get_to(v.vertical_pixels);
}

void to_json(nlohmann::json& j, ImageParams const& p)
{
    auto const& scalars = p.host_ref().scalars;
    j = nlohmann::json{
        CELER_JSON_PAIR(scalars, origin),
        CELER_JSON_PAIR(scalars, down),
        CELER_JSON_PAIR(scalars, right),
        CELER_JSON_PAIR(scalars, pixel_width),
        CELER_JSON_PAIR(scalars, dims),
    };
    j["_units"] = celeritas_units;
    j["int_size"] = sizeof(int);
}

//!@}
//---------------------------------------------------------------------------//
}  // namespace celeritas
