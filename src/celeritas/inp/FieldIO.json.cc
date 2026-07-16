//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/FieldIO.json.cc
//---------------------------------------------------------------------------//
#include "FieldIO.json.hh"

#include <variant>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/field/RZMapFieldInputIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo Move RZMap IO here

void to_json(nlohmann::json& j, NoField const&)
{
    j = {json_type_pair("none")};
}

void from_json(nlohmann::json const&, NoField&) {}

void to_json(nlohmann::json& j, UniformField const& v)
{
    //! \todo Write volumes
    j = {
        json_type_pair("uniform"),
        CELER_JSON_PAIR(v, units),
        CELER_JSON_PAIR(v, strength),
        CELER_JSON_PAIR(v, driver_options),
    };
}

void from_json(nlohmann::json const& j, UniformField& v)
{
    //! \todo Load volumes
    CELER_JSON_LOAD_OPTION(j, v, units);
    CELER_JSON_LOAD_OPTION(j, v, strength);
    CELER_JSON_LOAD_OPTION(j, v, driver_options);
}

void to_json(nlohmann::json& j, CylMapField const& v)
{
    j = {
        json_type_pair("cylmap"),
        CELER_JSON_PAIR(v, grid_r),
        CELER_JSON_PAIR(v, grid_phi),
        CELER_JSON_PAIR(v, grid_z),
        CELER_JSON_PAIR(v, field),
        CELER_JSON_PAIR(v, driver_options),
    };
}

void from_json(nlohmann::json const& j, CylMapField& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, grid_r);
    CELER_JSON_LOAD_REQUIRED(j, v, grid_phi);
    CELER_JSON_LOAD_REQUIRED(j, v, grid_z);
    CELER_JSON_LOAD_REQUIRED(j, v, field);
    CELER_JSON_LOAD_OPTION(j, v, driver_options);
}

template<class T>
void to_json(nlohmann::json& j, AxisGrid<T> const& v)
{
    j = {
        CELER_JSON_PAIR(v, min),
        CELER_JSON_PAIR(v, max),
        CELER_JSON_PAIR(v, num),
    };
}

template<class T>
void from_json(nlohmann::json const& j, AxisGrid<T>& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, min);
    CELER_JSON_LOAD_REQUIRED(j, v, max);
    CELER_JSON_LOAD_REQUIRED(j, v, num);
}

void to_json(nlohmann::json& j, CartMapField const& v)
{
    j = {
        json_type_pair("cartmap"),
        CELER_JSON_PAIR(v, x),
        CELER_JSON_PAIR(v, y),
        CELER_JSON_PAIR(v, z),
        CELER_JSON_PAIR(v, field),
        CELER_JSON_PAIR(v, driver_options),
    };
}

void from_json(nlohmann::json const& j, CartMapField& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, x);
    CELER_JSON_LOAD_REQUIRED(j, v, y);
    CELER_JSON_LOAD_REQUIRED(j, v, z);
    CELER_JSON_LOAD_REQUIRED(j, v, field);
    CELER_JSON_LOAD_OPTION(j, v, driver_options);
}

void to_json(nlohmann::json& j, Field const& v)
{
    j = std::visit([](auto const& f) { return nlohmann::json(f); }, v);
}

void from_json(nlohmann::json const& j, Field& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, none, NoField);
    CELER_JSON_LOAD_VARIANT(j, v, uniform, UniformField);
    CELER_JSON_LOAD_VARIANT(j, v, rzmap, RZMapField);
    CELER_JSON_LOAD_VARIANT(j, v, cylmap, CylMapField);
    CELER_JSON_LOAD_VARIANT(j, v, cartmap, CartMapField);
    CELER_VALIDATE(false, << "invalid Field input");
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
