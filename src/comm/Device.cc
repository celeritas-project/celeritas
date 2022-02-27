//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.cc
//---------------------------------------------------------------------------//
#include "Device.hh"

#include <cstdlib>
#include <iostream>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "base/device_runtime_api.h"
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
    if (!CELER_USE_DEVICE)
    {
        CELER_LOG(debug) << "Disabling GPU support since CUDA and HIP are "
                            "disabled";
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
    CELER_DEVICE_CALL_PREFIX(GetDeviceCount(&result));
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
        CELER_DEVICE_CALL_PREFIX(GetDevice(&cur_id));
        if (cur_id != device.device_id())
        {
            CELER_LOG(warning)
                << "CUDA active device ID unexpectedly changed from "
                << device.device_id() << " to " << cur_id;
            device = Device(cur_id);
        }
    }

#if CELER_USE_DEVICE && defined(_OPENMP)
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

#if CELER_USE_DEVICE
#    if CELERITAS_USE_CUDA
    cudaDeviceProp props;
#    elif CELERITAS_USE_HIP
    hipDeviceProp_t props;
#    endif

    CELER_DEVICE_CALL_PREFIX(GetDeviceProperties(&props, id));
    name_             = props.name;
    total_global_mem_ = props.totalGlobalMem;
    max_threads_      = props.maxThreadsPerMultiProcessor;
    warp_size_        = props.warpSize;
#    if CELERITAS_USE_HIP
    if (name_.empty())
    {
        // HIP has an extra field that seems to be used instead of the regular
        // props.name
        name_ = props.gcnArchName;
    }
#    endif

    extra_["clock_rate"]            = props.clockRate;
    extra_["multiprocessor_count"]  = props.multiProcessorCount;
    extra_["max_cache_size"]        = props.l2CacheSize;
    extra_["max_threads_per_block"] = props.maxThreadsPerBlock;
    extra_["memory_clock_rate"]     = props.memoryClockRate;
    extra_["regs_per_block"]        = props.regsPerBlock;
    extra_["shared_mem_per_block"]  = props.sharedMemPerBlock;
    extra_["total_const_mem"]       = props.totalConstMem;
#    if CELERITAS_USE_CUDA
    extra_["max_blocks_per_multiprocessor"] = props.maxBlocksPerMultiProcessor;
    extra_["regs_per_multiprocessor"]       = props.regsPerMultiprocessor;
#    endif
#endif

#if CELERITAS_USE_HIP && defined(__HIP_PLATFORM_AMD__)
    // AMD cards have 4 SIMD execution units per multiprocessor
    eu_per_mp_ = 4;
#elif CELERITAS_USE_CUDA || defined(__HIP_PLATFORM_NVIDIA__) \
    || defined(__HIP_PLATFORM_NVCC__)
    // CUDA: each streaming multiprocessor (MP) has one execution unit (EU)
    eu_per_mp_ = 1;
#endif
    CELER_ENSURE(*this);
    CELER_ENSURE(!name_.empty());
    CELER_ENSURE(total_global_mem_ > 0);
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
    CELER_DEVICE_CALL_PREFIX(SetDevice(device.device_id()));

    global_device() = std::move(device);

    // Call cudaFree to wake up the device, making other timers more accurate
    CELER_DEVICE_CALL_PREFIX(Free(nullptr));
    CELER_LOG(debug) << "Device initialization took " << get_time() << "s";
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
 * Increase CUDA stack size to enable complex geometries with VecGeom.
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
