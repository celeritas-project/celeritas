//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
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
 * be a singleton.
 */
class Device
{
  public:
    //!@{
    //! Type aliases
    using size_type = std::size_t;
    //!@}

  public:
    // Number of devices available on the local compute node
    static unsigned int num_devices();

    // Construct in round-robin fashion from an MPI communicator
    static Device from_round_robin(const Communicator& comm);

    //// CONSTRUCTORS ////

    // Construct an inactive device (disable celeritas CUDA calls)
    Device() = default;

    // Construct from device ID
    explicit Device(int id);

    //// ACCESSORS ////

    // Get the CUDA device ID
    inline int device_id() const;

    //! True if device is initialized
    explicit operator bool() const { return id_ >= 0; }

    //! Device name
    std::string name() const { return name_; }

    //! Total memory capacity (bytes)
    size_type total_global_mem() const { return total_global_mem_; }

    //! Maximum number of threads per multiprocessor
    int max_threads() const { return max_threads_; }

    //! Number of multiprocessors
    int num_multi_processors() const { return num_multi_processors_; }

    //! Number of threads per warp
    unsigned int warp_size() const { return warp_size_; }

    //! Default number of threads per block (TODO: make configurable?)
    unsigned int default_block_size() const { return 256; }

  private:
    int          id_                   = -1;
    std::string  name_                 = "<DISABLED>";
    size_type    total_global_mem_     = 0;
    int          max_threads_          = 0;
    int          num_multi_processors_ = 0;
    unsigned int warp_size_            = 0;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Global active device (default is inactive/false)
const Device& device();

// Activate the devie
void activate_device(Device&& device);

// Print device info
std::ostream& operator<<(std::ostream&, const Device&);

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
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
