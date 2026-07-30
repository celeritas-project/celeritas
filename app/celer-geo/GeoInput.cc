//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/GeoInput.cc
//---------------------------------------------------------------------------//
#include "GeoInput.hh"

#include "corecel/Types.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert a geometry string to an enum for JSON input.
 */
Geometry to_geometry(std::string const& s)
{
    static auto const from_string
        = StringEnumMapper<Geometry>::from_cstring_func(to_cstring, "geometry");
    return from_string(s);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a memspace string to an enum for JSON input.
 */
MemSpace to_memspace(std::string const& s)
{
    static auto const from_string
        = StringEnumMapper<MemSpace>::from_cstring_func(
            ::celeritas::to_cstring, "memspace");
    return from_string(s);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
#define GI_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, v, NAME)
#define GI_LOAD_REQUIRED(NAME) CELER_JSON_LOAD_REQUIRED(j, v, NAME)
#define GI_SAVE_NONZERO(NAME) CELER_JSON_SAVE_WHEN(j, v, NAME, v.NAME != 0)
#define GI_SAVE(NAME) CELER_JSON_SAVE(j, v, NAME)

void from_json(nlohmann::json const& j, ModelSetup& v)
{
    CELER_VALIDATE(j.is_object(),
                   << "input JSON for ModelSetup is not an object: '"
                   << j.dump() << '\'');
    GI_LOAD_OPTION(cuda_stack_size);
    GI_LOAD_OPTION(cuda_heap_size);
    GI_LOAD_REQUIRED(geometry_file);
    GI_LOAD_OPTION(perfetto_file);
}

void from_json(nlohmann::json const& j, TraceSetup& v)
{
    if (auto iter = j.find("geometry"); iter != j.end() && !iter->is_null())
    {
        v.geometry = to_geometry(iter->get<std::string>());
    }
    if (auto iter = j.find("memspace"); iter != j.end() && !iter->is_null())
    {
        v.memspace = to_memspace(iter->get<std::string>());
    }
    GI_LOAD_OPTION(volumes);
    GI_LOAD_REQUIRED(bin_file);
}

void to_json(nlohmann::json& j, ModelSetup const& v)
{
    GI_SAVE_NONZERO(cuda_stack_size);
    GI_SAVE_NONZERO(cuda_heap_size);
    GI_SAVE(geometry_file);
}

void to_json(nlohmann::json& j, ModelSetupOutput const& v)
{
    // Save base attributes
    j = static_cast<ModelSetup const&>(v);
    // Save versions
    GI_SAVE(version_string);
    GI_SAVE(version_hex);
}

void to_json(nlohmann::json& j, TraceSetup const& v)
{
    j["geometry"] = to_cstring(v.geometry);
    j["memspace"] = to_cstring(v.memspace);
    GI_SAVE(volumes);
    GI_SAVE(bin_file);
}

#undef GI_LOAD_OPTION
#undef GI_LOAD_REQUIRED
#undef GI_SAVE_NONZERO
#undef GI_SAVE

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
