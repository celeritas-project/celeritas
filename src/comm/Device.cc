//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.cc
//---------------------------------------------------------------------------//
#include "Device.hh"

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif
#ifdef _OPENMP
#    include <omp.h>
#endif

#include <cstdlib>
#include <iostream>
#include "base/Assert.hh"
#include "base/Stopwatch.hh"
#include "Communicator.hh"
#include "Environment.hh"
#include "Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
int determine_num_devices()
{
    if (!CELERITAS_USE_CUDA)
    {
        CELER_LOG(debug) << "Disabling GPU support since CUDA is disabled";
        return 0;
    }

    if (!celeritas::getenv("CELER_DISABLE_DEVICE").empty())
    {
        CELER_LOG(info)
            << "Disabling GPU support since the 'CELER_DISABLE_DEVICE' "
               "environment variable is present and non-empty";
        return 0;
    }

    int result = -1;
    CELER_CUDA_CALL(cudaGetDeviceCount(&result));
    if (result == 0)
    {
        CELER_LOG(warning) << "Disabling GPU support since no CUDA devices "
                              "are present";
    }

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Active CUDA device for Celeritas calls on the local thread/process.
 *
 * \todo This function is not thread-friendly. It assumes distributed memory
 * parallelism with one device assigned per process. See
 * https://github.com/celeritas-project/celeritas/pull/149#discussion_r577997723
 * and
 * https://github.com/celeritas-project/celeritas/pull/149#discussion_r578000062
 */
Device& global_device()
{
    static Device device;
    if (CELERITAS_DEBUG && device)
    {
        // Check that CUDA and Celeritas device IDs are consistent
        int cur_id = -1;
        CELER_CUDA_CALL(cudaGetDevice(&cur_id));
        if (cur_id != device.device_id())
        {
            CELER_LOG(warning)
                << "CUDA active device ID unexpectedly changed from "
                << device.device_id() << " to " << cur_id;
            device = Device(cur_id);
        }
    }

#if CELERITAS_USE_CUDA && defined(_OPENMP)
    if (omp_get_num_threads() > 1)
    {
        CELER_NOT_IMPLEMENTED("OpenMP support with CUDA");
    }
#endif

    return device;
}

} // namespace

//---------------------------------------------------------------------------//
// DEVICE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the number of available devices.
 *
 * This is nonzero if and only if CUDA support is built-in, if at least one
 * CUDA-capable device is present, and if the 'CELER_DISABLE_DEVICE'
 * environment variable is not set.
 */
int Device::num_devices()
{
    static const int result = determine_num_devices();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize device in a round-robin fashion from a communicator.
 */
Device Device::from_round_robin(const Communicator& comm)
{
    int num_devices = Device::num_devices();
    if (num_devices == 0)
    {
        // Null device
        return {};
    }

    return Device(comm.rank() % num_devices);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a device ID.
 */
Device::Device(int id) : id_(id)
{
    CELER_EXPECT(id >= 0 && id < Device::num_devices());

#if CELERITAS_USE_CUDA
    cudaDeviceProp props;
    CELER_CUDA_CALL(cudaGetDeviceProperties(&props, id));
    name_                 = props.name;
    total_global_mem_     = props.totalGlobalMem;
    max_threads_          = props.maxThreadsPerMultiProcessor;
    num_multi_processors_ = props.multiProcessorCount;
    warp_size_            = props.warpSize;
#endif

    CELER_ENSURE(*this);
    CELER_ENSURE(!name_.empty());
    CELER_ENSURE(total_global_mem_ > 0);
    CELER_ENSURE(max_threads_ > 0);
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the shared default device.
 */
const Device& device()
{
    return global_device();
}

//---------------------------------------------------------------------------//
/*!
 * Activate the given device.
 *
 * The given device must be set (true result) unless no device has yet been
 * enabled -- this allows Device::from_round_robin to create "null" devices
 * when CUDA is disabled.
 */
void activate_device(Device&& device)
{
    CELER_EXPECT(device || !global_device());

    if (!device)
        return;

    Stopwatch get_time;

    // Set device based on communicator, and call cudaFree to wake up the
    // device
    CELER_LOG_LOCAL(debug) << "Initializing '" << device.name() << "', ID "
                           << device.device_id() << " of "
                           << Device::num_devices();
    CELER_CUDA_CALL(cudaSetDevice(device.device_id()));

    global_device() = std::move(device);

    // Call cudaFree to wake up the device, making other timers more accurate
    CELER_CUDA_CALL(cudaFree(nullptr));
    CELER_LOG(debug) << "CUDA initialization took " << get_time() << "s";
}

//---------------------------------------------------------------------------//
/*!
 * Print device info.
 */
std::ostream& operator<<(std::ostream& os, const Device& d)
{
    if (d)
    {
        os << "<device " << d.device_id() << ": " << d.name() << ">";
    }
    else
    {
        os << "<inactive device>";
    }
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Increase CUDA stack size to enable complex geometries.
 *
 * For the cms2018.gdml detector geometry, the default stack size is too small,
 * and a limit of 32768 is recommended.
 */
void set_cuda_stack_size(int limit)
{
    CELER_EXPECT(limit > 0);
    CELER_EXPECT(celeritas::device());
    CELER_CUDA_CALL(cudaDeviceSetLimit(cudaLimitStackSize, limit));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
