//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TypesIO.json.cc
//---------------------------------------------------------------------------//
#include "TypesIO.json.hh"

#include <unordered_map>

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read track order options from JSON.
 */
void from_json(nlohmann::json const& j, TrackOrder& value)
{
    static auto const from_string
        = StringEnumMapper<TrackOrder>::from_cstring_func(to_cstring,
                                                          "track order");
    auto&& jstr = j.get<std::string>();
    try
    {
        value = from_string(jstr);
    }
    catch (RuntimeError const& e)
    {
        static std::unordered_map<std::string, TrackOrder> old_names{
            {"unsorted", TrackOrder::none},
            {"partition_charge", TrackOrder::init_charge},
            {"shuffled", TrackOrder::reindex_shuffle},
            {"partition_status", TrackOrder::reindex_status},
            {"sort_along_step_action", TrackOrder::reindex_along_step_action},
            {"sort_step_limit_action", TrackOrder::reindex_step_limit_action},
            {"sort_action", TrackOrder::reindex_both_action},
            {"sort_particle_type", TrackOrder::reindex_particle_type},
        };
        if (auto iter = old_names.find(jstr); iter != old_names.end())
        {
            value = iter->second;
            CELER_LOG(warning)
                << "Deprecated track order label '" << jstr << "': use '"
                << to_cstring(value) << "' instead";
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read interpolation method from JSON.
 */
void from_json(nlohmann::json const& j, InterpolationType& value)
{
    static auto const from_string
        = StringEnumMapper<InterpolationType>::from_cstring_func(
            to_cstring, "interpolation");
    value = from_string(j.get<std::string>());
}

//---------------------------------------------------------------------------//
/*!
 * Write track order options to JSON.
 */
void to_json(nlohmann::json& j, TrackOrder const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
/*!
 * Write interpolation method to JSON.
 */
void to_json(nlohmann::json& j, InterpolationType const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
