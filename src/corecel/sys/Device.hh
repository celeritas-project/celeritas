//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>  // IWYU pragma: keep
#include <map>
#include <string>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage attributes of the GPU.
 *
 * CUDA/HIP translation table:
 *
 * CUDA/NVIDIA    | HIP/AMD        | Description
 * -------------- | -------------- | -----------------
 * thread         | work item      | individual local work element
 * warp           | wavefront      | "vectorized thread" operating in lockstep
 * block          | workgroup      | group of threads able to sync
 * multiprocessor | compute unit   | hardware executing one or more blocks
 * multiprocessor | execution unit | hardware executing one or more warps
 *
 * Each block/workgroup operates on the same hardware (compute unit) until
 * completion. Similarly, a warp/wavefront is tied to a single execution
 * unit. Each compute unit can execute one or more blocks: the higher the
 * number of blocks resident, the more latency can be hidden.
 *
 * \todo The active CUDA device is a global property---so this should probably
 * be a singleton, or we could use lower-level API calls.
 *
 * \note The current multithreading/multiprocess model is intended to have one
 * GPU serving multiple CPU threads simultaneously, and one MPI process per
 * GPU. If we need to add multiple GPUs per process, then \c global_device will
 * go away in favor of a thread-independent model.
 */
class Device
{
  public:
    //!@{
    //! Type aliases
    using MapStrInt = std::map<std::string, int>;
    //!@}

  public:
    // Number of devices available on the local compute node (0 if disabled)
    static int num_devices();

    // Whether verbose messages and error checking are enabled
    static bool debug();

    //// CONSTRUCTORS ////

    // Construct an inactive device (disable celeritas CUDA calls)
    Device() = default;

    // Construct from device ID
    explicit Device(int id);

    //// ACCESSORS ////

    // Get the device ID
    inline int device_id() const;

    //! True if device is initialized
    explicit operator bool() const { return id_ >= 0; }

    //! Device name
    std::string name() const { return name_; }

    //! Total memory capacity (bytes)
    std::size_t total_global_mem() const { return total_global_mem_; }

    //! Maximum number of threads per block (for launch limits)
    int max_threads_per_block() const { return max_threads_per_block_; }

    //! Maximum number of threads per block (for launch limits)
    int max_blocks_per_grid() const { return max_blocks_per_grid_; }

    //! Maximum number of concurrent threads per compute unit (for occupancy)
    int max_threads_per_cu() const { return max_threads_per_cu_; }

    //! Number of threads per warp
    unsigned int threads_per_warp() const { return threads_per_warp_; }

    //! Number of execution units per compute unit (1 for NVIDIA, 4 for AMD)
    unsigned int eu_per_cu() const { return eu_per_cu_; }

    //! Default number of threads per block
    unsigned int default_block_size() const { return default_block_size_; }

    //! Additional potentially interesting diagnostics
    MapStrInt const& extra() const { return extra_; }

  private:
    int id_ = -1;
    std::string name_ = "<DISABLED>";
    std::size_t total_global_mem_ = 0;
    int max_threads_per_block_ = 0;
    int max_blocks_per_grid_ = 0;
    int max_threads_per_cu_ = 0;
    unsigned int threads_per_warp_ = 0;
    unsigned int eu_per_cu_ = 0;
    unsigned int default_block_size_ = 256u;
    MapStrInt extra_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Global active device (default is inactive/false)
Device const& device();

// Activate the device
void activate_device(Device&& device);

// Print device info
std::ostream& operator<<(std::ostream&, Device const&);

// Increase CUDA stack size
void set_cuda_stack_size(int limit);

// Increase CUDA HEAP size
void set_cuda_heap_size(int limit);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the CUDA device ID, if active.
 */
int Device::device_id() const
{
    CELER_EXPECT(*this);
    return id_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
