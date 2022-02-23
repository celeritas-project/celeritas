//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <celeritas_config.h>
#include <algorithm>
#include <iosfwd>
#include <unordered_map>
#include <type_traits>
#include <vector>
#include "base/Assert.hh"
#include "base/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Properties for a single kernel.
struct KernelProperties
{
    std::string  name;
    unsigned int block_size = 0;
    unsigned int device_id  = 0;

    int         num_regs  = 0; //!< Number of 32-bit registers per thread
    std::size_t const_mem = 0; //!< Amount of constant memory (per thread) [b]
    std::size_t local_mem = 0; //!< Amount of local memory (per thread) [b]
    double      occupancy = 0; //!< Max blocks per multiprocessor

    unsigned int num_launches    = 0; //!< Number of times launched
    unsigned int max_num_threads = 0; //!< Highest number of threads used
};

//---------------------------------------------------------------------------//
/*!
 * Program diagnostics helper class.
 *
 * There should generally be only a single instance of this, accessible through
 * the \c kernel_diagnostics helper function.
 */
class KernelDiagnostics
{
  public:
    //!@{
    //! Type aliases
    using key_type        = OpaqueId<struct Kernel>;
    using value_type      = KernelProperties;
    using const_reference = const value_type&;
    using size_type       = key_type::size_type;
    //!@}

  public:
    // Construct without any data
    KernelDiagnostics() = default;

    // Register a kernel, gathering diagnostics if needed
    template<class F>
    inline key_type
    insert(F* func_ptr, const char* name, unsigned int block_size);

    //! Number of kernel diagnostics available
    size_type size() const { return values_.size(); }

    // Get the kernel diagnostics for a given ID
    inline const_reference at(key_type id) const;

    //// DIAGNOSTICS ////

    // Mark that a kernel was launched with this many threads
    inline void launch(key_type key, unsigned int num_threads);

  private:
    // Map of kernel function address to kernel IDs
    std::unordered_map<std::uintptr_t, key_type> keys_;

    // Kernel diagnostics
    std::vector<value_type> values_;

    //// HELPER FUNCTIONS ////
    void log_launch(value_type& diag, unsigned int num_threads);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Global reference to diagnostics
KernelDiagnostics& kernel_diagnostics();

// Write the diagnostics to a stream
std::ostream& operator<<(std::ostream&, const KernelDiagnostics&);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the kernel diagnostics for a given ID.
 */
auto KernelDiagnostics::at(key_type key) const -> const_reference
{
    CELER_EXPECT(key < this->size());
    return values_[key.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Mark that a kernel was launched with this many threads.
 */
void KernelDiagnostics::launch(key_type key, unsigned int num_threads)
{
    CELER_EXPECT(key < this->size());
    value_type& diag = values_[key.get()];
    ++diag.num_launches;
    diag.max_num_threads = std::max(num_threads, diag.max_num_threads);
#if CELERITAS_DEBUG
    this->log_launch(diag, num_threads);
#endif
}

#ifdef CELER_DEVICE_SOURCE
//---------------------------------------------------------------------------//
/*!
 * Register the given __global__ kernel function.
 *
 * This can only be called from CUDA code.
 */
template<class F>
inline auto
KernelDiagnostics::insert(F* func, const char* name, unsigned int block_size)
    -> key_type
{
    static_assert(std::is_function<F>::value,
                  "KernelDiagnostics must be called with a function object, "
                  "not a function pointer or anything else.");
    auto iter_inserted = keys_.insert(
        {reinterpret_cast<std::uintptr_t>(func), key_type{this->size()}});
    if (CELER_UNLIKELY(iter_inserted.second))
    {
        // First time this kernel was added
        value_type diag;

        const Device& device = celeritas::device();
        diag.device_id       = device.device_id();
        diag.name            = name;
        diag.block_size      = block_size;

        CELER_DEVICE_PREFIX(FuncAttributes) attr;
        CELER_DEVICE_CALL_PREFIX(
            FuncGetAttributes(&attr, reinterpret_cast<const void*>(func)));
        diag.num_regs  = attr.numRegs;
        diag.const_mem = attr.constSizeBytes;
        diag.local_mem = attr.localSizeBytes;

        std::size_t dynamic_smem_size = 0;
        int         num_blocks        = 0;
        CELER_DEVICE_CALL_PREFIX(OccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks, func, diag.block_size, dynamic_smem_size));
        diag.occupancy = static_cast<double>(num_blocks * diag.block_size)
                         / device.max_threads();

        values_.push_back(std::move(diag));
    }

    CELER_ENSURE(keys_.size() == values_.size());
    CELER_ENSURE(iter_inserted.first->second < this->size());
    return iter_inserted.first->second;
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
