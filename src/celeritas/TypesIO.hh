//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TypesIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
inline void from_json(nlohmann::json const& j, TrackOrder& value)
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
        std::unordered_map<std::string, TrackOrder> old_names{
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
 * Write options to JSON.
 */
inline void to_json(nlohmann::json& j, TrackOrder const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
