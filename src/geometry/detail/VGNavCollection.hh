//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavCollection.hh
//---------------------------------------------------------------------------//
#pragma once

#ifndef __CUDA_ARCH__
#    include <memory>
#endif
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/navigation/NavStatePool.h>
#include "base/Assert.hh"
#include "base/OpaqueId.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Collection-like container for managing VecGeom navigation states.
 *
 * Since reference and value all behave differently for host and device, we
 * only *declare* the class, and provide specializations for each type. The
 * specializations are also explicitly declared before their definitions, since
 * many of the definitions may not be available depending on which compiler or
 * which compilation phase is active.
 */
template<Ownership W, MemSpace M>
struct VGNavCollection;

template<>
struct VGNavCollection<Ownership::value, MemSpace::host>;
template<>
struct VGNavCollection<Ownership::value, MemSpace::device>;
template<>
struct VGNavCollection<Ownership::reference, MemSpace::host>;
template<>
struct VGNavCollection<Ownership::reference, MemSpace::device>;

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
// HOST MEMSPACE
//---------------------------------------------------------------------------//
/*!
 * Manage a single navigation state in host memory.
 *
 * Since navigation states are allocated on the heap, and don't have a default
 * contructor, we must use a `unique_ptr` to manage its memory.
 */
template<>
struct VGNavCollection<Ownership::value, MemSpace::host>
{
    using NavState = vecgeom::cxx::NavigationState;

    std::unique_ptr<NavState> nav_state;

    // Resize with a number of states (must be 1)
    void resize(int max_depth, size_type size);
    // Whether the collection is assigned
    explicit operator bool() const { return static_cast<bool>(nav_state); }
};

//---------------------------------------------------------------------------//
/*!
 * Reference a host-owned navigation state.
 */
template<>
struct VGNavCollection<Ownership::reference, MemSpace::host>
{
    using NavState = vecgeom::cxx::NavigationState;

    NavState* ptr = nullptr;

    // Obtain reference from host memory
    void operator=(VGNavCollection<Ownership::value, MemSpace::host>& other);
    // Get the navigation state for a given thread
    NavState& at(int, ThreadId id) const;
    //! True if the collection is assigned/valiid
    explicit operator bool() const { return static_cast<bool>(ptr); }
};
#endif

//---------------------------------------------------------------------------//
// DEVICE MEMSPACE

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Delete a VecGeom pool.
 *
 * Due to VecGeom macros, the definition of this function can only be compiled
 * from a .cc file.
 */
struct NavStatePoolDeleter
{
    using arg_type = vecgeom::cxx::NavStatePool*;
    void operator()(arg_type) const;
};

//---------------------------------------------------------------------------//
/*!
 * Manage a pool of device-side geometry states.
 *
 * Construction and destruction of the NavStatePool has to be in a host
 * compilation unit due to VecGeom macro magic. We hide this class to keep
 * NavStatePool and smart pointer usage from the NVCC device compiler.
 */
template<>
struct VGNavCollection<Ownership::value, MemSpace::device>
{
    using UPNavStatePool
        = std::unique_ptr<vecgeom::cxx::NavStatePool, NavStatePoolDeleter>;

    UPNavStatePool pool;
    void*          ptr  = nullptr;
    int            max_depth = 0;
    size_type      size = 0;

    // Resize based on geometry params and state size
    void resize(int max_depth, size_type size);
    //! True if the collection is assigned/valid
    explicit CELER_FUNCTION operator bool() const { return ptr; }
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Reference on-device memory owned by VGNavCollection<value, device>.
 *
 * The NavStatePool underpinning the storage returns a void pointer that must
 * be manually manipulated to get a single state pointer. The max_depth
 * argument must be the same as the GeoParams.
 */
template<>
struct VGNavCollection<Ownership::reference, MemSpace::device>
{
    using NavState = vecgeom::cuda::NavigationState;

    void*     ptr       = nullptr;
    int       max_depth = 0;
    size_type size      = 0;

    // Assign from device value
    void operator=(VGNavCollection<Ownership::value, MemSpace::device>& other);
    // Get the navigation state for the given thread
    inline CELER_FUNCTION NavState& at(int max_depth, ThreadId thread) const;

    //! True if the collection is assigned/valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ptr && size > 0 && max_depth > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Get the navigation state at the given thread.
 *
 * The max_depth_param is used for error checking against the allocated
 * max_depth.
 */
CELER_FUNCTION auto VGNavCollection<Ownership::reference, MemSpace::device>::at(
    int max_depth_param, ThreadId thread) const -> NavState&
{
    CELER_EXPECT(this->ptr);
    CELER_EXPECT(thread < this->size);
    CELER_EXPECT(max_depth_param == this->max_depth);
#ifdef __NVCC__
    // This code only compiles when run through CUDA so it must be escaped.
    char* result = reinterpret_cast<char*>(this->ptr);
    result += NavState::SizeOfInstanceAlignAware(max_depth_param)
              * thread.get();
    return *reinterpret_cast<NavState*>(result);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
