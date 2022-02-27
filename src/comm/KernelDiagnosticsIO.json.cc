//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnosticsIO.json.cc
//---------------------------------------------------------------------------//
#include "KernelDiagnosticsIO.json.hh"

#include "base/Range.hh"

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
            {"block_size", diag.block_size},
            {"num_regs", diag.num_regs},
            {"const_mem", diag.const_mem},
            {"local_mem", diag.local_mem},
            {"num_launches", diag.num_launches},
            {"max_num_threads", diag.max_num_threads},
            {"max_threads_per_block", diag.max_threads_per_block},
            {"max_blocks_per_mp", diag.max_blocks_per_mp},
            {"max_warps_per_mp", diag.max_warps_per_mp},
            {"occupancy", diag.occupancy},
        }));
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
