//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <atomic>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"

#include "KernelAttributes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct KernelProfiling
{
    //!< Number of times launched
    std::atomic<int> num_launches{0};
    //!< Number of threads integrated over all launches
    std::atomic<long long> accum_threads{0};

    // Increment atomic counters given the number of threads
    inline void log_launch(int num_threads);
};

//---------------------------------------------------------------------------//
struct KernelMetadata
{
    std::string      name;
    KernelAttributes attributes;
    KernelProfiling  profiling;
};

//! Ordered identifiers for registered kernels
using KernelId = OpaqueId<KernelMetadata>;

//---------------------------------------------------------------------------//
/*!
 * Keep track of kernels and launches.
 *
 * Every "insert" creates a unique \c KernelMetadata entry in a thread-safe
 * fashion (in case multiple threads are launching kernels for the first time).
 * Thus every kernel added to the registry needs a \c static local data (i.e.,
 * \c KernelParamCalculator) to track whether the kernel has been added and to
 * keep a reference to the returned profiling data counter. Kernels are always
 * added sequentially and can never be removed from the registry once added.
 * Kernels that share the same name will create independent entries!
 *
 * \warning Until all kernels have been inserted, the \c num_kernels and
 * \c kernel accessors are not safe to
 * use. Accessing by ID is expected to be done well after all insertions have
 * been completed.
 *
 * This class has a thread-safe *insert* method because it's meant to be shared
 * across multiple threads when running.
 */
class KernelRegistry
{
  public:
    // Whether profiling metrics (launch count, max threads) are collected
    static bool profiling();

    // Construct without any data
    KernelRegistry() = default;

    //// CONSTRUCTION ////

    // Register a kernel and return optional reference to profiling info
    KernelProfiling* insert(const char* name, KernelAttributes&& attrs);

    //// ACCESSORS ////

    //! Number of kernel diagnostics available (not thread-safe)
    KernelId::size_type num_kernels() const { return kernels_.size(); }

    // Access kernel data for a single kernel (not thread-safe)
    inline const KernelMetadata& kernel(KernelId id) const;

  private:
    using UPKM = std::unique_ptr<KernelMetadata>;

    std::mutex        insert_mutex_;
    std::vector<UPKM> kernels_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Globally shared registry of kernels for end-of-program diagnostics
KernelRegistry& kernel_registry();

// Write kernel statistics to a stream
std::ostream& operator<<(std::ostream& os, const KernelMetadata& md);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Accumulate counters for a kernel launch.
 */
void KernelProfiling::log_launch(int num_threads)
{
    CELER_EXPECT(num_threads > 0);

    // Increment launches by 1 and thread count by num_threads.
    // We don't care in what order these values are written.
    this->num_launches.fetch_add(1, std::memory_order_relaxed);
    this->accum_threads.fetch_add(num_threads, std::memory_order_relaxed);
}

//---------------------------------------------------------------------------//
/*!
 * Get the kernel metadata for a given ID.
 */
auto KernelRegistry::kernel(KernelId k) const -> const KernelMetadata&
{
    CELER_EXPECT(k < this->num_kernels());
    return *kernels_[k.unchecked_get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
