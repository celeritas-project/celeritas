//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.cc
//---------------------------------------------------------------------------//
#include "KernelDiagnostics.hh"

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

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
/*!
 * Add kernel diagnostics.
 */
void KernelDiagnostics::push_back_kernel(value_type                     diag,
                                         CELER_MAYBE_UNUSED const void* func)
{
#if CELERITAS_USE_CUDA
    const Device& device = celeritas::device();
    diag.device_id       = device.device_id();

    cudaFuncAttributes attr;
    CELER_CUDA_CALL(cudaFuncGetAttributes(&attr, func));
    diag.num_regs  = attr.numRegs;
    diag.const_mem = attr.constSizeBytes;
    diag.local_mem = attr.localSizeBytes;

    std::size_t dynamic_smem_size = 0;
    int         num_blocks        = 0;
    CELER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, func, diag.block_size, dynamic_smem_size));
    diag.occupancy = static_cast<double>(num_blocks * diag.block_size)
                     / device.max_threads();
#else
    CELER_NOT_CONFIGURED("CUDA");
#endif
    values_.push_back(std::move(diag));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
