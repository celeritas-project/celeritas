//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Device.cc
//---------------------------------------------------------------------------//
#include "Device.hh"

#include <iostream>  // IWYU pragma: keep
#include <limits>
#include <mutex>
#include <utility>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#if CELERITAS_USE_OPENMP
#    include <omp.h>
#endif

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"

#include "Environment.hh"
#include "MpiCommunicator.hh"
#include "Stream.hh"

#include "detail/StreamStorage.hh"

#if CELERITAS_USE_CUDA
#    define CELER_DEVICE_SUPPORTS_MEMPOOL 1
#elif CELERITAS_USE_HIP       \
    && (HIP_VERSION_MAJOR > 5 \
        || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 2))
#    define CELER_DEVICE_SUPPORTS_MEMPOOL 1
#else
#    define CELER_DEVICE_SUPPORTS_MEMPOOL 0
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Active CUDA device for Celeritas calls on the local process.
 *
 * \todo This function assumes distributed memory parallelism with one device
 * assigned per process. See
 * https://github.com/celeritas-project/celeritas/pull/149#discussion_r577997723
 * and
 * https://github.com/celeritas-project/celeritas/pull/149#discussion_r578000062
 *
 * The device should be *activated* by the main thread, and \c
 * activate_device_local should be called on other threads to set up the
 * local CUDA context.
 */
Device& global_device()
{
    static Device device;
    if (device && Device::debug())
    {
        // Check that CUDA and Celeritas device IDs are consistent
        int cur_id = -1;
        CELER_DEVICE_CALL_PREFIX(GetDevice(&cur_id));
        CELER_ASSERT(cur_id == device.device_id());
    }

    return device;
}

//---------------------------------------------------------------------------//
}  // namespace

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
    static int const result = [] {
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
            CELER_LOG(warning)
                << "Disabling GPU support since no CUDA devices "
                   "are present";
        }

        CELER_ENSURE(result >= 0);
        return result;
    }();
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
    static bool const result = [] {
        if constexpr (CELERITAS_DEBUG)
        {
            return true;
        }
        return !celeritas::getenv("CELER_DEBUG_DEVICE").empty();
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a device ID.
 */
Device::Device(int id) : id_{id}, streams_{new detail::StreamStorage{}}
{
    CELER_EXPECT(id >= 0 && id < Device::num_devices());

    CELER_LOG_LOCAL(debug) << "Constructing device ID " << id;

#if CELER_USE_DEVICE
#    if CELERITAS_USE_CUDA
    using DevicePropT = cudaDeviceProp;
#    elif CELERITAS_USE_HIP
    using DevicePropT = hipDeviceProp_t;
#    endif

    DevicePropT props;
    CELER_DEVICE_CALL_PREFIX(GetDeviceProperties(&props, id));

    name_ = props.name;
    total_global_mem_ = props.totalGlobalMem;
    max_threads_per_block_ = props.maxThreadsDim[0];
    max_blocks_per_grid_ = props.maxGridSize[0];
    max_threads_per_cu_ = props.maxThreadsPerMultiProcessor;
    threads_per_warp_ = props.warpSize;
    can_map_host_memory_ = props.canMapHostMemory != 0;
#    if CELERITAS_USE_HIP
    if (name_.empty())
    {
        // The name attribute may be missing? (true for ROCm 4.5.0/gfx90a), so
        // assume the name can be extracted from the GCN arch:
        // "gfx90a:sramecc+:xnack-" (SRAM ECC and XNACK are memory related
        // flags )
        std::string gcn_arch_name = props.gcnArchName;
        auto pos = gcn_arch_name.find(':');
        if (pos != std::string::npos)
        {
            gcn_arch_name.erase(pos);
            name_ = std::move(gcn_arch_name);
        }
    }
#    endif

    extra_["clock_rate"] = props.clockRate;
    extra_["multiprocessor_count"] = props.multiProcessorCount;
    extra_["max_cache_size"] = props.l2CacheSize;
    extra_["memory_clock_rate"] = props.memoryClockRate;
    extra_["regs_per_block"] = props.regsPerBlock;
    extra_["shared_mem_per_block"] = props.sharedMemPerBlock;
    extra_["total_const_mem"] = props.totalConstMem;
    extra_["capability_major"] = props.major;
    extra_["capability_minor"] = props.minor;
#    if CELERITAS_USE_CUDA
#        if CUDART_VERSION >= 11000
    extra_["max_blocks_per_multiprocessor"] = props.maxBlocksPerMultiProcessor;
#        endif
    extra_["regs_per_multiprocessor"] = props.regsPerMultiprocessor;
#    endif

    // Save for comparison to CMake configuration
    capability_ = 10 * props.major + props.minor;

    // Save for possible block size initialization
    max_threads_per_block_ = props.maxThreadsPerBlock;
#endif

#if CELER_DEVICE_SUPPORTS_MEMPOOL
    auto threshold = std::numeric_limits<uint64_t>::max();
    if (std::string var = celeritas::getenv("CELER_MEMPOOL_RELEASE_THRESHOLD");
        !var.empty())
    {
        threshold = std::stoul(var);
    }
    CELER_DEVICE_PREFIX(MemPool_t) mempool;
    CELER_DEVICE_CALL_PREFIX(DeviceGetDefaultMemPool(&mempool, id_));
    CELER_DEVICE_CALL_PREFIX(MemPoolSetAttribute(
        mempool, CELER_DEVICE_PREFIX(MemPoolAttrReleaseThreshold), &threshold));
#endif

    // See DeviceRuntimeApi.hh
    eu_per_cu_ = CELER_EU_PER_CU;

    CELER_ENSURE(*this);
    CELER_ENSURE(!name_.empty());
    CELER_ENSURE(total_global_mem_ > 0);
    CELER_ENSURE(max_threads_per_block_ > 0 && max_blocks_per_grid_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Number of streams allocated.
 */
StreamId::size_type Device::num_streams() const
{
    if (!streams_)
        return 0;
    return streams_->size();
}

//---------------------------------------------------------------------------//
/*!
 * Allocate the given number of streams.
 *
 * If no streams have been created, the default stream will be used.
 */
void Device::create_streams(unsigned int num_streams) const
{
    CELER_EXPECT(*this);
    CELER_EXPECT(streams_);

    CELER_LOG(info) << "Creating " << num_streams << " device streams";
    *streams_ = detail::StreamStorage(num_streams);
}

//---------------------------------------------------------------------------//
/*!
 * Access a stream.
 *
 * This returns the default stream if no streams were allocated.
 */
Stream& Device::stream(StreamId id) const
{
    CELER_EXPECT(streams_);

    return streams_->get(id);
}

//---------------------------------------------------------------------------//
/*!
 * External deleter for stream storage.
 *
 * This is for the PIMPL idiom with unique pointers.
 */
void Device::StreamStorageDeleter::operator()(detail::StreamStorage* p) noexcept
{
    delete p;
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the shared default device.
 */
Device const& device()
{
    return global_device();
}

//---------------------------------------------------------------------------//
/*!
 * Activate the global celeritas device.
 *
 * The given device must be set (true result) unless no device has yet been
 * enabled -- this allows \c make_device to create "null" devices
 * when CUDA is disabled.
 *
 * This function may be called once only, because the global device propagates
 * into local states (e.g. where memory is allocated) all over Celeritas.
 */
void activate_device(Device&& device)
{
    static std::mutex m;
    std::lock_guard<std::mutex> scoped_lock{m};
    Device& d = global_device();
    CELER_VALIDATE(!d,
                   << "celeritas::activate_device may be called only once per "
                      "application");

    if (!device)
        return;

    // Check capability version against cmake variable; rough but better than
    // nothing! CMake format: "native" or "70-real 72-virtual" or "35;50;72" or
    // for HIP, "gfx90a"
    std::string_view const arch{celeritas_gpu_architectures};
    if (arch != "native"
        && arch.find(std::to_string(device.capability())) == std::string::npos)
    {
        constexpr auto gpu_str = CELERITAS_USE_CUDA  ? "CUDA"
                                 : CELERITAS_USE_HIP ? "HIP"
                                                     : "";
        CELER_LOG(warning)
            << "Device '" << device.name() << "' has " << gpu_str
            << " compute capability of " << device.capability()
            << ", but Celeritas was compiled with CMAKE_" << gpu_str
            << "_ARCHITECTURES=\"" << celeritas_gpu_architectures
            << "\": code may mysteriously die at runtime";
    }

    CELER_LOG_LOCAL(debug) << "Initializing '" << device.name() << "', ID "
                           << device.device_id() << " of "
                           << Device::num_devices();

    ScopedTimeLog scoped_time(&self_logger(), 1.0);
    CELER_DEVICE_CALL_PREFIX(SetDevice(device.device_id()));
    d = std::move(device);

    // Call cudaFree to wake up the device, making other timers more accurate
    CELER_DEVICE_CALL_PREFIX(Free(nullptr));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the first device if available, when not using MPI.
 */
void activate_device()
{
    return activate_device(celeritas::comm_world());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize device in a round-robin fashion from a communicator.
 */
void activate_device(MpiCommunicator const& comm)
{
    int num_devices = Device::num_devices();
    if (num_devices > 0)
    {
        return activate_device(Device(comm.rank() % num_devices));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Call cudaSetDevice using the existing device, for thread-local safety.
 *
 * See
 * https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs
 *
 * \pre activate_device was called or no device is intended to be used
 */
void activate_device_local()
{
    Device& d = global_device();
    if (d)
    {
        CELER_LOG_LOCAL(debug) << "Activating device " << d.device_id();
        CELER_DEVICE_CALL_PREFIX(SetDevice(d.device_id()));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print device info.
 */
std::ostream& operator<<(std::ostream& os, Device const& d)
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
 * and a limit of 8K is recommended with debugging disabled (and up to 32K if
 * debugging is enabled).
 */
void set_cuda_stack_size(int limit)
{
    CELER_EXPECT(limit > 0);
    if (!celeritas::device())
    {
        CELER_LOG(warning) << "Ignoring call to set_cuda_stack_size: no "
                              "device is available";
        return;
    }
    if constexpr (CELERITAS_USE_CUDA)
    {
        CELER_LOG(debug) << "Setting CUDA stack size to " << limit << "B";
    }
    CELER_CUDA_CALL(cudaDeviceSetLimit(cudaLimitStackSize, limit));
}

//---------------------------------------------------------------------------//
/*!
 * Increase CUDA heap size to enable complex geometries with VecGeom.
 *
 * For the cms-hllhc.gdml detector geometry, the 8MB default heap size is too
 * small, and a new size as high as 33554432 (=32MB) has run successfully.
 * This should be increased as necessary, but avoid setting it too high.
 */
void set_cuda_heap_size(int limit)
{
    CELER_EXPECT(limit > 0);
    if (!celeritas::device())
    {
        CELER_LOG(warning) << "Ignoring call to set_cuda_stack_size: no "
                              "device is available";
        return;
    }
    if constexpr (CELERITAS_USE_CUDA)
    {
        CELER_LOG(debug) << "Setting CUDA heap size to " << limit << "B";
    }
    CELER_CUDA_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
