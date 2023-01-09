//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelRegistryIO.json.cc
//---------------------------------------------------------------------------//
#include "KernelRegistryIO.json.hh"

#include <atomic>
#include <string>

#include "corecel/cont/Range.hh"
#include "corecel/sys/KernelAttributes.hh"

#include "KernelRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write kernel metadata out to JSON.
 */
void to_json(nlohmann::json& j, const KernelRegistry& kr)
{
    const bool write_profiling = KernelRegistry::profiling();

    j = nlohmann::json::array();
    for (auto kernel_id : range(KernelId{kr.num_kernels()}))
    {
        const auto& md = kr.kernel(kernel_id);
        j.emplace_back(nlohmann::json::object({
            {"name", md.name},
            {"threads_per_block", md.attributes.threads_per_block},
            {"num_regs", md.attributes.num_regs},
            {"const_mem", md.attributes.const_mem},
            {"local_mem", md.attributes.local_mem},
            {"max_threads_per_block", md.attributes.max_threads_per_block},
            {"max_blocks_per_cu", md.attributes.max_blocks_per_cu},
            {"max_warps_per_eu", md.attributes.max_warps_per_eu},
            {"occupancy", md.attributes.occupancy},
        }));
        if (write_profiling)
        {
            j.back()["num_launches"]
                = static_cast<int>(md.profiling.num_launches);
            j.back()["accum_threads"]
                = static_cast<int>(md.profiling.accum_threads);
        }
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
