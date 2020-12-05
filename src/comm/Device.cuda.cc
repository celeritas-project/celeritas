//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.cc
//---------------------------------------------------------------------------//
#include "Device.hh"

#include <cstdlib>
#include <cuda_runtime_api.h>
#include "base/Assert.hh"
#include "comm/Logger.hh"

namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
bool determine_device_enable()
{
    const char* disable = std::getenv("CELER_DISABLE_DEVICE");
    if (disable && disable[0] != '\0')
    {
        CELER_LOG(info)
            << "Disabling GPU support since the 'CELER_DISABLE_DEVICE' "
               "environment variable is present and non-empty";
        return false;
    }
    int num_devices = -1;
    cudaGetDeviceCount(&num_devices);
    if (num_devices <= 0)
    {
        CELER_LOG(warning) << "Disabling GPU support since no CUDA devices "
                              "are present";
        return false;
    }
    return true;
}
} // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Determine whether the CUDA device is enabled.
 *
 * This is true if and only if CUDA support is built-in, if at least one
 * CUDA-capable device is present, and if the 'CELER_DISABLE_DEVICE'
 * environment variable is not set.
 */
bool is_device_enabled()
{
    static const bool result = determine_device_enable();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize device in a round-robin fashion from a communicator.
 */
void initialize_device(const Communicator& comm)
{
    if (!is_device_enabled())
        return;

    // Get number of devices
    int num_devices = -1;
    CELER_CUDA_CALL(cudaGetDeviceCount(&num_devices));
    CHECK(num_devices > 0);

    // Set device based on communicator
    int device_id = comm.rank() % num_devices;
    CELER_LOG_LOCAL(debug) << "Initializing device ID " << device_id << " of "
                           << num_devices;
    CELER_CUDA_CALL(cudaSetDevice(device_id));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
