//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceIO.json.cc
//---------------------------------------------------------------------------//
#include "DeviceIO.json.hh"

#include "celeritas_config.h"
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
            {"max_threads_per_cu", d.max_threads_per_cu()},
            {"threads_per_warp", d.threads_per_warp()},
            {"eu_per_cu", d.eu_per_cu()},
            {"default_block_size", d.default_block_size()},
        };

#if CELERITAS_USE_CUDA
        j["platform"] = "cuda";
#elif CELERITAS_USE_HIP
        j["platform"] = "hip";
#endif

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
