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
 * This class has a thread-safe *insert* method because it's meant to be shared
 * across multiple threads when running. Since accessing by ID is expected to
 * be done well after all insertions have been completed, the \c size and \c at
 * accessors are *not* thread safe.
 */
class KernelRegistry
{
  public:
    //!@{
    //! \name Type aliases
    using value_type      = KernelMetadata;
    using const_reference = const KernelMetadata&;
    using key_type        = KernelId;
    using size_type       = KernelId::size_type;
    //!@}

  public:
    // Whether profiling metrics (launch count, max threads) are collected
    static bool profiling();

    // Construct without any data
    KernelRegistry() = default;

    // Register a kernel and return optional reference to profiling info
    KernelProfiling* insert(const char* name, KernelAttributes&& attrs);

    //! Number of kernel diagnostics available
    size_type size() const { return kernels_.size(); }

    // Access kernel data for a single kernel
    inline const_reference at(key_type id) const;

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
auto KernelRegistry::at(key_type key) const -> const_reference
{
    CELER_EXPECT(key < this->size());
    return *kernels_[key.unchecked_get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
