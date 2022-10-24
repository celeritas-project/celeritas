//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelDiagnostics.cc
//---------------------------------------------------------------------------//
#include "KernelDiagnostics.hh"

#include <atomic>
#include <iostream>
#include <mutex>

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"

#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Try adding a new kernel in a thread-safe fashion.
 */
auto KernelDiagnostics::threadsafe_insert_new(std::uintptr_t fptr)
    -> InsertResult
{
    static std::mutex           insert_mutex;
    std::lock_guard<std::mutex> scoped_lock{insert_mutex};
    return keys_.insert({fptr, key_type{this->size()}});
}

//---------------------------------------------------------------------------//
/*!
 * In debug mode, log a message about an impending launch.
 *
 * In debug mode performance isn't critical, so it's OK to have unused log
 * messages.
 */
void KernelDiagnostics::log_launch(value_type& diag, unsigned int num_threads)
{
    static std::atomic<int> remaining_log_msg{32};
    int                     remaining = --remaining_log_msg;
    if (remaining > 0)
    {
        CELER_LOG(debug)
            << "Launching '" << diag.name << "' on " << diag.threads_per_block
            << " threads per block with " << num_threads << " total threads";
    }
    else if (remaining == 0)
    {
        CELER_LOG(debug) << "(Launch log limit reached: suppressing further "
                            "output)";
    }
}

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
            "  name: \""            << diag.name              << "\",\n"
            "  threads_per_block: " << diag.threads_per_block << ",\n"
            "  num_regs: "          << diag.num_regs          << ",\n"
            "  const_mem: "         << diag.const_mem         << ",\n"
            "  local_mem: "         << diag.local_mem         << ",\n"
            "  occupancy: "         << diag.occupancy         << ",\n"
            "  num_launches: "      << diag.num_launches      << ",\n"
            "  max_num_threads: "   << diag.max_num_threads   << "\n"
            "}";
        // clang-format on
    }
    os << "])";
    return os;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
