//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class Communicator;

//---------------------------------------------------------------------------//
/*!
 * Manage attributes of the GPU.
 *
 * \todo The active CUDA device is a global property -- so this should probably
 * be a singleton, or we could use lower-level API calls.
 */
class Device
{
  public:
    //!@{
    //! Type aliases
    using MapStrInt = std::map<std::string, int>;
    //!@}

  public:
    // Number of devices available on the local compute node
    static int num_devices();

    // Construct in round-robin fashion from an MPI communicator
    static Device from_round_robin(const Communicator& comm);

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

    //! Maximum number of threads per multiprocessor (for occupancy)
    int max_threads() const { return max_threads_; }

    //! Number of threads per warp
    unsigned int warp_size() const { return warp_size_; }

    //! Number of execution units per multiprocessor (1 for NVIDIA, 4 for AMD)
    unsigned int eu_per_mp() const { return eu_per_mp_; }

    //! Default number of threads per block (TODO: make configurable)
    unsigned int default_block_size() const { return 256u; }

    //! Additional potentially interesting diagnostics
    const MapStrInt& extra() const { return extra_; }

  private:
    int          id_               = -1;
    std::string  name_             = "<DISABLED>";
    std::size_t  total_global_mem_ = 0;
    int          max_threads_      = 0;
    unsigned int warp_size_        = 0;
    unsigned int eu_per_mp_        = 0;
    MapStrInt    extra_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Global active device (default is inactive/false)
const Device& device();

// Activate the device
void activate_device(Device&& device);

// Print device info
std::ostream& operator<<(std::ostream&, const Device&);

// Increase CUDA stack size
void set_cuda_stack_size(int limit);

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
} // namespace celeritas
