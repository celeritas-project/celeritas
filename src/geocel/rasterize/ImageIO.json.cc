//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageIO.json.cc
//---------------------------------------------------------------------------//
#include "ImageIO.json.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/math/ArrayOperators.hh"
#include "geocel/detail/LengthUnits.hh"

#include "Image.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
#define IM_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, v, NAME)
#define IM_LOAD_REQUIRED(NAME) CELER_JSON_LOAD_REQUIRED(j, v, NAME)

void from_json(nlohmann::json const& j, ImageInput& v)
{
    IM_LOAD_REQUIRED(lower_left);
    IM_LOAD_REQUIRED(upper_right);
    IM_LOAD_REQUIRED(rightward);
    IM_LOAD_REQUIRED(vertical_pixels);
    IM_LOAD_OPTION(horizontal_divisor);

    real_type length{lengthunits::centimeter};
    if (auto iter = j.find("_units"); iter != j.end())
    {
        switch (to_unit_system(iter->get<std::string>()))
        {
            case UnitSystem::cgs:
                length = lengthunits::centimeter;
                break;
            case UnitSystem::si:
                length = lengthunits::meter;
                break;
            case UnitSystem::clhep:
                length = lengthunits::millimeter;
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    if (length != 1)
    {
        v.lower_left *= length;
        v.upper_right *= length;
    }
}

void to_json(nlohmann::json& j, ImageInput const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, lower_left),
        CELER_JSON_PAIR(v, upper_right),
        CELER_JSON_PAIR(v, rightward),
        CELER_JSON_PAIR(v, vertical_pixels),
        CELER_JSON_PAIR(v, horizontal_divisor),
        {"_units", to_cstring(UnitSystem::native)},
    };
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
        CELER_JSON_PAIR(scalars, max_length),
        {"_units", to_cstring(UnitSystem::native)},
    };
}

#undef IM_LOAD_OPTION
#undef IM_LOAD_REQUIRED
//!@}
//---------------------------------------------------------------------------//
}  // namespace celeritas
