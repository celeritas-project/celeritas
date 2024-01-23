//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    using MemQuantity = KibiBytes;
    j = nlohmann::json::object();
#define MRIO_MEM_OUT(MEMBER)                                  \
    do                                                        \
    {                                                         \
        if (entry.MEMBER.value() > 0)                         \
        {                                                     \
            j[#MEMBER] = value_as<MemQuantity>(entry.MEMBER); \
        }                                                     \
    } while (false)
    MRIO_MEM_OUT(cpu_delta);
    MRIO_MEM_OUT(cpu_hwm);
    MRIO_MEM_OUT(gpu_delta);
    MRIO_MEM_OUT(gpu_usage);
#undef MRIO_MEM_OUT
    if (!j.empty())
    {
        j["_units"] = MemQuantity::unit_type::label();
    }
    if (entry.parent_index)
    {
        j["parent_index"] = entry.parent_index.unchecked_get();
    }
    j["label"] = entry.label;
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
