//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceIO.json.cc
//---------------------------------------------------------------------------//
#include "DeviceIO.json.hh"

#include <map>

#include "corecel/Config.hh"
#include "corecel/DeviceRuntimeApi.hh"

#include "Device.hh"

namespace celeritas
{
#define CELER_DIO_PAIR(ATTR) {#ATTR, d.ATTR()}
//---------------------------------------------------------------------------//
/*!
 * Write device diagnostics out to JSON.
 */
void to_json(nlohmann::json& j, Device const& d)
{
    if (d)
    {
        j = nlohmann::json{
            CELER_DIO_PAIR(device_id),
            CELER_DIO_PAIR(name),
            CELER_DIO_PAIR(total_global_mem),
            CELER_DIO_PAIR(max_threads_per_block),
            CELER_DIO_PAIR(max_blocks_per_grid),
            CELER_DIO_PAIR(max_threads_per_cu),
            CELER_DIO_PAIR(threads_per_warp),
            CELER_DIO_PAIR(eu_per_cu),
            CELER_DIO_PAIR(capability),
            CELER_DIO_PAIR(can_map_host_memory),
            // Static data
            CELER_DIO_PAIR(debug),
            CELER_DIO_PAIR(num_devices),
        };

        j["platform"] = CELER_DEVICE_PLATFORM_UPPER_STR;
        j["async"] = Device::async();

        for (auto const& kv : d.extra())
        {
            j[kv.first] = kv.second;
        }
    }
    else
    {
        j = nlohmann::json{};
    }
}

#undef CELER_DIO_PAIR

//---------------------------------------------------------------------------//
}  // namespace celeritas
