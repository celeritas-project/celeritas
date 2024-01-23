//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelRegistryIO.json.cc
//---------------------------------------------------------------------------//
#include "KernelRegistryIO.json.hh"

#include <atomic>
#include <string>

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "corecel/sys/KernelAttributes.hh"

#include "KernelRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write one kernel's metadata to JSON.
 */
void to_json(nlohmann::json& j, KernelAttributes const& attrs)
{
    j = {
        {"threads_per_block", attrs.threads_per_block},
        {"num_regs", attrs.num_regs},
        {"const_mem", attrs.const_mem},
        {"local_mem", attrs.local_mem},
        {"max_threads_per_block", attrs.max_threads_per_block},
        {"max_blocks_per_cu", attrs.max_blocks_per_cu},
        {"max_warps_per_eu", attrs.max_warps_per_eu},
        {"occupancy", attrs.occupancy},
        {"heap_size", attrs.heap_size},
        {"print_buffer_size", attrs.print_buffer_size},
    };
    if constexpr (CELERITAS_USE_CUDA)
    {
        j["stack_size"] = attrs.stack_size;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write kernel metadata out to JSON.
 */
void to_json(nlohmann::json& j, KernelRegistry const& kr)
{
    bool const write_profiling = KernelRegistry::profiling();

    j = nlohmann::json::array();
    for (auto kernel_id : range(KernelId{kr.num_kernels()}))
    {
        auto const& md = kr.kernel(kernel_id);
        j.push_back(md.attributes);
        j.back()["name"] = md.name;
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
}  // namespace celeritas
