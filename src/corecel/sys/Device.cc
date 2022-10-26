//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Device.cc
//---------------------------------------------------------------------------//
#include "Device.hh"

#include <cstdlib>
#include <iostream>

#include "celeritas_config.h"
#if CELERITAS_USE_OPENMP
#    include <omp.h>
#endif

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"

#include "Environment.hh"

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
 * Whether to check and warn about inconsistent CUDA/Celeritas device.
 */
bool determine_debug()
{
    if (CELERITAS_DEBUG)
    {
        return true;
    }
    return !celeritas::getenv("CELER_DEBUG_DEVICE").empty();
}

//---------------------------------------------------------------------------//
/*!
 * Active CUDA device for Celeritas calls on the local process.
 *
 * \todo This function assumes distributed memory parallelism with one device
 * assigned per process. See
 * https://github.com/celeritas-project/celeritas/pull/149#discussion_r577997723
 * and
 * https://github.com/celeritas-project/celeritas/pull/149#discussion_r578000062
 */
Device& global_device()
{
    static Device device;
    if (device && Device::debug())
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

    return device;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the number of available devices.
 *
 * This is nonzero if and only if CUDA support is built-in, if at least one
 * CUDA-capable device is present, and if the \c CELER_DISABLE_DEVICE
 * environment variable is not set.
 */
int Device::num_devices()
{
    static const int result = determine_num_devices();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Whether verbose messages and error checking are enabled.
 *
 * This is true if \c CELERITAS_DEBUG is set *or* if the \c CELER_DEBUG_DEVICE
 * environment variable exists and is not empty.
 */
bool Device::debug()
{
    static const bool result = determine_debug();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a device ID.
 */
Device::Device(int id) : id_(id)
{
    CELER_EXPECT(id >= 0 && id < Device::num_devices());

    unsigned int max_threads_per_block = 0;
#if CELER_USE_DEVICE
#    if CELERITAS_USE_CUDA
    cudaDeviceProp props;
#    elif CELERITAS_USE_HIP
    hipDeviceProp_t props;
#    endif

    CELER_DEVICE_CALL_PREFIX(GetDeviceProperties(&props, id));
    name_               = props.name;
    total_global_mem_   = props.totalGlobalMem;
    max_threads_per_cu_ = props.maxThreadsPerMultiProcessor;
    threads_per_warp_   = props.warpSize;
#    if CELERITAS_USE_HIP
    if (name_.empty())
    {
        // The name attribute may be missing? (true for ROCm 4.5.0/gfx90a), so
        // assume the name can be extracted from the GCN arch:
        // "gfx90a:sramecc+:xnack-" (SRAM ECC and XNACK are memory related
        // flags )
        std::string gcn_arch_name = props.gcnArchName;
        auto        pos           = gcn_arch_name.find(':');
        if (pos != std::string::npos)
        {
            gcn_arch_name.erase(pos);
            name_ = std::move(gcn_arch_name);
        }
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
    extra_["capability_major"]      = props.major;
    extra_["capability_minor"]      = props.minor;
#    if CELERITAS_USE_CUDA
#        if CUDART_VERSION >= 11000
    extra_["max_blocks_per_multiprocessor"] = props.maxBlocksPerMultiProcessor;
#        endif
    extra_["regs_per_multiprocessor"] = props.regsPerMultiprocessor;
#    endif

    // Save for possible block size initialization
    max_threads_per_block = props.maxThreadsPerBlock;
#endif

    // See device_runtime_api.h
    eu_per_cu_ = CELER_EU_PER_CU;

    // Set default block size from environment
    const std::string& bsize_str = celeritas::getenv("CELER_BLOCK_SIZE");
    if (!bsize_str.empty())
    {
        default_block_size_ = std::stoi(bsize_str);
        CELER_VALIDATE(default_block_size_ >= threads_per_warp_
                           && default_block_size_ <= max_threads_per_block,
                       << "Invalid block size: number of threads must be in ["
                       << threads_per_warp_ << ", " << max_threads_per_block
                       << "]");
        CELER_VALIDATE(default_block_size_ % threads_per_warp_ == 0,
                       << "Invalid block size: number of threads must be "
                          "evenly divisible by "
                       << threads_per_warp_);
    }

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

    CELER_LOG_LOCAL(debug) << "Initializing '" << device.name() << "', ID "
                           << device.device_id() << " of "
                           << Device::num_devices();
    ScopedTimeLog scoped_time;
    CELER_DEVICE_CALL_PREFIX(SetDevice(device.device_id()));
    global_device() = std::move(device);

    // Call cudaFree to wake up the device, making other timers more accurate
    CELER_DEVICE_CALL_PREFIX(Free(nullptr));
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
