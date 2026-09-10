//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelRegistry.cc
//---------------------------------------------------------------------------//
#include "KernelRegistry.hh"

#include <iostream>  // IWYU pragma: keep
#include <utility>

#include "corecel/Config.hh"

#include "Environment.hh"
#include "KernelAttributes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether to record potentially expensive kernel profiling information.
 *
 * This is true if \c CELERITAS_DEBUG is set *or* if the \c
 * CELER_PROFILE_DEVICE environment variable exists and is not empty.
 */
bool KernelRegistry::profiling()
{
    static bool const result = [] {
        return getenv_flag("CELER_PROFILE_DEVICE", CELERITAS_DEBUG).value;
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Add a new kernel definition to the list
 */
auto KernelRegistry::insert(std::string_view name, KernelAttributes&& attrs)
    -> KernelProfiling*
{
    // Create metadata for this kernel
    auto kmd = std::make_unique<KernelMetadata>();
    kmd->name = name;
    kmd->attributes = std::move(attrs);

    // Save a pointer to the profiling data only if profiling is enabled
    KernelProfiling* result = KernelRegistry::profiling() ? &kmd->profiling
                                                          : nullptr;

    // Move the unique pointer onto the back of the kernel vector in a
    // thread-safe manner.
    std::lock_guard<std::mutex> scoped_lock{kernels_mutex_};
    kernels_.emplace_back(std::move(kmd));

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Number of kernel diagnostics available.
 */
KernelId::size_type KernelRegistry::num_kernels() const
{
    // Lock while calculating the vector size.
    std::lock_guard<std::mutex> scoped_lock{kernels_mutex_};
    return kernels_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the kernel metadata for a given ID.
 */
auto KernelRegistry::kernel(KernelId k) const -> KernelMetadata const&
{
    CELER_EXPECT(k < this->num_kernels());
    // Lock while accessing the vector; the reference itself is safe.
    std::lock_guard<std::mutex> scoped_lock{kernels_mutex_};
    return *kernels_[k.unchecked_get()];
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
std::ostream& operator<<(std::ostream& os, KernelMetadata const& md)
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
}  // namespace celeritas
