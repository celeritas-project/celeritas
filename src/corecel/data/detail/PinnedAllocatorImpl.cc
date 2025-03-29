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
namespace
{
bool enable_pinned()
{
    static bool const result = [] { return Device::num_devices() > 0; }();
    return result;
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Allocate and construct space for \c n objects of size \c sizof_t.
 *
 * If any devices are available at the first call, use pinned memory.
 * Otherwise, use standard allocation for the rest of the program lifetime.
 */
void* malloc_pinned(std::size_t n, std::size_t sizeof_t)
{
    CELER_EXPECT(n != 0);
    CELER_EXPECT(sizeof_t != 0);
    if (n > std::numeric_limits<std::size_t>::max() / sizeof_t)
        throw std::bad_array_new_length();

    void* p{nullptr};
    if (enable_pinned())
    {
        // NOTE: CUDA and HIP have a different API signature!
#if CELERITAS_USE_CUDA
        CELER_DEVICE_API_CALL(HostAlloc(
            &p, n * sizeof_t, CELER_DEVICE_API_SYMBOL(HostAllocDefault)));
#elif CELERITAS_USE_HIP
        CELER_DEVICE_API_CALL(HostMalloc(
            &p, n * sizeof_t, CELER_DEVICE_API_SYMBOL(HostMallocDefault)));
#else
        CELER_ASSERT_UNREACHABLE();
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
 */
void free_pinned(void* p) noexcept
{
    if (enable_pinned())
    {
        try
        {
            // NOTE: API calls differ between CUDA and HIP
#if CELERITAS_USE_CUDA
            CELER_DEVICE_API_CALL(FreeHost(p));
#elif CELERITAS_USE_HIP
            CELER_DEVICE_API_CALL(HostFree(p));
#else
            CELER_ASSERT_UNREACHABLE();
#endif
        }
        catch (std::exception const& e)
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
