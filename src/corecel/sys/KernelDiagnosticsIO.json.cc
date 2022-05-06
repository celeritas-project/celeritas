//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelDiagnosticsIO.json.cc
//---------------------------------------------------------------------------//
#include "KernelDiagnosticsIO.json.hh"

#include "corecel/cont/Range.hh"

#include "KernelDiagnostics.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write kernel diagnostics out to JSON.
 */
void to_json(nlohmann::json& j, const KernelDiagnostics& kd)
{
    j = nlohmann::json::array();
    for (auto kernel_idx : range(kd.size()))
    {
        const auto& diag = kd.at(KernelDiagnostics::key_type{kernel_idx});
        j.emplace_back(nlohmann::json::object({
            {"name", diag.name},
            {"threads_per_block", diag.threads_per_block},
            {"num_regs", diag.num_regs},
            {"const_mem", diag.const_mem},
            {"local_mem", diag.local_mem},
            {"num_launches", diag.num_launches},
            {"max_num_threads", diag.max_num_threads},
            {"max_threads_per_block", diag.max_threads_per_block},
            {"max_blocks_per_cu", diag.max_blocks_per_cu},
            {"max_warps_per_eu", diag.max_warps_per_eu},
            {"occupancy", diag.occupancy},
        }));
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
