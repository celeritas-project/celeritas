//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MemRegistryIO.json.cc
//---------------------------------------------------------------------------//
#include "MemRegistryIO.json.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/QuantityIO.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write one kernel's metadata to JSON.
 *
 * If the registry is dumped before a "scoped memory" saves the memory results,
 * no entries will be written. Parent indices are only written for child nodes.
 */
void to_json(nlohmann::json& j, MemUsageEntry const& entry)
{
    j = {
        {"label", entry.label},
    };
    if (entry.parent_index)
    {
        j["parent_index"] = entry.parent_index.unchecked_get();
    }
    if (entry.cpu_delta.value() > 0)
    {
        j["cpu_delta"] = entry.cpu_delta;
    }
    if (entry.cpu_hwm.value() > 0)
    {
        j["cpu_hwm"] = entry.cpu_hwm;
    }
    if (entry.gpu_delta.value() > 0)
    {
        j["gpu_delta"] = entry.gpu_delta;
    }
    if (entry.gpu_usage.value() > 0)
    {
        j["gpu_usage"] = entry.gpu_usage;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write kernel metadata out to JSON.
 */
void to_json(nlohmann::json& j, MemRegistry const& mr)
{
    j = nlohmann::json::array();
    for (auto mem_id : range(MemUsageId{mr.size()}))
    {
        j.push_back(mr.get(mem_id));
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
