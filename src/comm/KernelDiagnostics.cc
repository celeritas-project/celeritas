//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.cc
//---------------------------------------------------------------------------//
#include "KernelDiagnostics.hh"

#include <iostream>
#include "base/Macros.hh"
#include "base/Range.hh"
#include "comm/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Global reference to shared Celeritas kernel diagnostics.
 */
KernelDiagnostics& kernel_diagnostics()
{
    static KernelDiagnostics result;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Write the diagnostics to a stream.
 */
std::ostream& operator<<(std::ostream& os, const KernelDiagnostics& kd)
{
    os << "KernelDiagnostics([";
    for (auto kernel_idx : range(kd.size()))
    {
        if (kernel_idx > 0)
        {
            os << ',';
        }

        const auto& diag = kd.at(KernelDiagnostics::key_type{kernel_idx});
        // clang-format off
        os << "{\n"
            "  name: \""          << diag.name            << "\",\n"
            "  block_size: "      << diag.block_size      << ",\n"
            "  num_regs: "        << diag.num_regs        << ",\n"
            "  const_mem: "       << diag.const_mem       << ",\n"
            "  local_mem: "       << diag.local_mem       << ",\n"
            "  occupancy: "       << diag.occupancy       << ",\n"
            "  num_launches: "    << diag.num_launches    << ",\n"
            "  max_num_threads: " << diag.max_num_threads << "\n"
            "}";
        // clang-format on
    }
    os << "])";
    return os;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
