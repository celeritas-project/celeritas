//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceIO.json.cc
//---------------------------------------------------------------------------//
#include "DeviceIO.json.hh"

#include "comm/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write device diagnostics out to JSON.
 */
void to_json(nlohmann::json& j, const Device& d)
{
    if (d)
    {
        j = nlohmann::json{
            {"device_id", d.device_id()},
            {"name", d.name()},
            {"total_global_mem", d.total_global_mem()},
            {"max_threads", d.max_threads()},
            {"eu_per_mp", d.eu_per_mp()},
            {"warp_size", d.warp_size()},
            {"default_block_size", d.default_block_size()},
        };

        for (const auto& kv : d.extra())
        {
            j[kv.first] = kv.second;
        }
    }
    else
    {
        j = nlohmann::json{};
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
