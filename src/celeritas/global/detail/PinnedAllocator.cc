//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/PinnedAllocator.cc
//---------------------------------------------------------------------------//
#include "PinnedAllocator.hh"

#include <limits>
#include <new>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/DetectorSteps.hh"

namespace celeritas
{
namespace detail
{
template<class T>
T* PinnedAllocator<T>::allocate(std::size_t n)
{
    CELER_EXPECT(n != 0);
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::bad_array_new_length();

#if CELER_USE_DEVICE
    void* p{nullptr};
    // CUDA and HIP currently have a different API to allocate pinned host
    // memory
#    if CELERITAS_USE_CUDA
    CELER_DEVICE_CALL_PREFIX(
        HostAlloc(&p, n * sizeof(T), CELER_DEVICE_PREFIX(HostAllocDefault)));
#    elif CELERITAS_USE_HIP
    CELER_DEVICE_CALL_PREFIX(
        HostMalloc(&p, n * sizeof(T), CELER_DEVICE_PREFIX(HostMallocDefault)));
#    endif
    if (p)
#else
    if (auto p = ::operator new(n * sizeof(T)); p)
#endif
    {
        return static_cast<T*>(p);
    }

    throw std::bad_alloc();
}

template<class T>
void PinnedAllocator<T>::deallocate(T* p, std::size_t) noexcept
{
#if CELER_USE_DEVICE
    try
    {
#    if CELERITAS_USE_CUDA
        CELER_DEVICE_CALL_PREFIX(FreeHost(p));
#    elif CELERITAS_USE_HIP
        CELER_DEVICE_CALL_PREFIX(HostFree(p));
#    endif
    }
    catch (RuntimeError const& e)
    {
        CELER_LOG(debug) << "While freeing pinned host memory: " << e.what();
    }

#else
    ::operator delete(p);
#endif
}
//---------------------------------------------------------------------------//
// Explicit instantiations
template struct PinnedAllocator<real_type>;
template struct PinnedAllocator<size_type>;
template struct PinnedAllocator<Real3>;
template struct PinnedAllocator<DetectorStepPointOutput::Energy>;
template struct PinnedAllocator<DetectorId>;
template struct PinnedAllocator<TrackId>;
template struct PinnedAllocator<EventId>;
template struct PinnedAllocator<ParticleId>;
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
