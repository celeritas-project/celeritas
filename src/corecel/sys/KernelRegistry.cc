//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelRegistry.cc
//---------------------------------------------------------------------------//
#include "KernelRegistry.hh"

#include <iostream>

#include "corecel/Macros.hh"
#include "corecel/sys/Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//
bool determine_profiling()
{
    if (CELERITAS_DEBUG)
    {
        return true;
    }
    return !celeritas::getenv("CELER_PROFILE_DEVICE").empty();
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether to record potentially expensive kernel profiling information.
 *
 * This is true if \c CELERITAS_DEBUG is set *or* if the \c
 * CELER_PROFILE_DEVICE environment variable exists and is not empty.
 */
bool KernelRegistry::profiling()
{
    static const bool result = determine_profiling();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Try adding a new kernel in a thread-safe fashion.
 */
auto KernelRegistry::insert(const char* name, KernelAttributes&& attrs)
    -> KernelProfiling*
{
    // Create metadata for this kernel
    auto kmd        = std::make_unique<KernelMetadata>();
    kmd->name       = name;
    kmd->attributes = std::move(attrs);

    // Save a pointer to the profiling data only if profiling is enabled
    KernelProfiling* result = KernelRegistry::profiling() ? &kmd->profiling
                                                          : nullptr;

    // Move the unique pointer onto the back of the kernel vector in a
    // thread-safe manner.
    std::lock_guard<std::mutex> scoped_lock{insert_mutex_};
    kernels_.emplace_back(std::move(kmd));

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Globally shared kernel registry.
 */
KernelRegistry& kernel_registry()
{
    static KernelRegistry kr;
    return kr;
}

//---------------------------------------------------------------------------//
/*!
 * Write kernel metadata to a stream.
 */
std::ostream& operator<<(std::ostream& os, const KernelMetadata& md)
{
    // clang-format off
    os << "{\n"
        "  name: \""            << md.name              << "\",\n"
        "  num_regs: "          << md.attributes.num_regs          << ",\n"
        "  const_mem: "         << md.attributes.const_mem         << ",\n"
        "  local_mem: "         << md.attributes.local_mem         << ",\n"
        "  threads_per_block: " << md.attributes.threads_per_block << ",\n"
        "  occupancy: "         << md.attributes.occupancy;
    if (KernelRegistry::profiling())
    {
        os << ",\n"
            "  num_launches: "  << md.profiling.num_launches << ",\n"
            "  accum_threads: " << md.profiling.accum_threads;
    }
    os << "\n}";
    // clang-format on
    return os;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
