//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/PinnedAllocatorImpl.cc
//---------------------------------------------------------------------------//
#include "PinnedAllocatorImpl.hh"

#include <limits>
#include <new>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Allocate and construct space for \c n objects of size \c sizof_t.
 *
 * If any devices are available, use pinned memory. Otherwise, use standard
 * allocation.
 */
void* malloc_pinned(std::size_t n, std::size_t sizeof_t)
{
    CELER_EXPECT(n != 0);
    CELER_EXPECT(sizeof_t != 0);
    if (n > std::numeric_limits<std::size_t>::max() / sizeof_t)
        throw std::bad_array_new_length();

    void* p{nullptr};
    if (Device::num_devices() > 0)
    {
        // CUDA and HIP currently have a different API to allocate pinned host
        // memory
#if CELERITAS_USE_CUDA
        CELER_CUDA_CALL(cudaHostAlloc(
            &p, n * sizeof_t, CELER_DEVICE_PREFIX(HostAllocDefault)));
#elif CELERITAS_USE_HIP
        CELER_HIP_CALL(hipHostMalloc(
            &p, n * sizeof_t, CELER_DEVICE_PREFIX(HostMallocDefault)));
#endif
    }
    else
    {
        p = ::operator new(n * sizeof_t);
    }

    if (!p)
    {
        throw std::bad_alloc();
    }

    return p;
}

//---------------------------------------------------------------------------//
/*!
 * Free allocated memory.
 *
 * Because \c Device::num_devices is static, this will always be symmetric
 * with the \c PinnedAllocator::allocate call.
 */
void free_pinned(void* p) noexcept
{
    if (Device::num_devices() > 0)
    {
        try
        {
            // NOTE that Free/Host switch places in the two languages
#if CELERITAS_USE_CUDA
            CELER_CUDA_CALL(cudaFreeHost(p));
#elif CELERITAS_USE_HIP
            CELER_HIP_CALL(hipHostFree(p));
#endif
        }
        catch (RuntimeError const& e)
        {
            CELER_LOG(debug)
                << "While freeing pinned host memory: " << e.what();
        }
    }
    else
    {
        ::operator delete(p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
