//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MemRegistryIO.json.cc
//---------------------------------------------------------------------------//
#include "MemRegistryIO.json.hh"

#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write one kernel's metadata to JSON.
 */
void to_json(nlohmann::json& j, MemUsageEntry const& entry)
{
    j = {
        {"label", entry.label},
        {"parent_index",
         entry.parent_index
             ? static_cast<int>(entry.parent_index.unchecked_get())
             : -1},
        {"cpu_delta", entry.cpu_delta.value()},
        {"cpu_hwm", entry.cpu_hwm.value()},
    };
    if (entry.gpu_hwm.value() != 0)
    {
        j["gpu_delta"] = entry.gpu_delta.value();
        j["gpu_hwm"] = entry.gpu_hwm.value();
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
