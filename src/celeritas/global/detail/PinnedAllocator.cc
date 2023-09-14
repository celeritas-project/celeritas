//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/PinnedAllocator.cc
//---------------------------------------------------------------------------//
#include "PinnedAllocator.hh"

#include <limits>
#include <new>
#if !CELER_USE_DEVICE
#    include <memory>
#endif

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
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
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::bad_array_new_length();

#if CELER_USE_DEVICE
    void* p{nullptr};
    CELER_DEVICE_CALL_PREFIX(MallocHost(&p, n * sizeof(T)));
    if (p)
#else
    if (void* p = std::malloc(n); p)
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
    // Not using CELER_DEVICE_CALL_PREFIX, must be noexcept
    CELER_DEVICE_PREFIX(FreeHost(p));
#else
    std::free(p);
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
