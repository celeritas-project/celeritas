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
#include "base/Assert.hh"
#include "base/OpaqueId.hh"
#include "base/Types.hh"

namespace vecgeom
{
#ifndef __NVCC__
inline
#endif
    namespace cxx
{
class NavStatePool;
} // namespace cxx
} // namespace vecgeom

// CUDA declarations aren't reachable from C++ host-compiled code, and the
// reverse, so we must fudge by forward-declaring here as though
// NavigationState is an untemplated class, not a type alias, for each unused
// memspace.
namespace vecgeom
{
#ifdef __NVCC__
namespace cxx
#else
namespace cuda
#endif
{
class NavigationState;
}
} // namespace vecgeom

namespace celeritas
{
namespace detail
{
template<Ownership W, MemSpace M>
struct VGNavCollection;

// Declare specializations because each one is different and may not be
// defined based on active compiler/phase
template<>
struct VGNavCollection<Ownership::value, MemSpace::host>;
template<>
struct VGNavCollection<Ownership::value, MemSpace::device>;
template<>
struct VGNavCollection<Ownership::reference, MemSpace::host>;
template<>
struct VGNavCollection<Ownership::reference, MemSpace::device>;

#ifndef __CUDA_ARCH__
template<>
struct VGNavCollection<Ownership::value, MemSpace::host>
{
    using NavState = vecgeom::cxx::NavigationState;

    std::unique_ptr<NavState> nav_state;

    void resize(int max_depth, int size);

    explicit operator bool() const { return static_cast<bool>(nav_state); }
};

template<>
struct VGNavCollection<Ownership::reference, MemSpace::host>
{
    using NavState = vecgeom::cxx::NavigationState;

    NavState* ptr = nullptr;

    VGNavCollection& operator=(const VGNavCollection&) = default;

    void operator=(VGNavCollection<Ownership::value, MemSpace::host>& other);

    NavState& at(int, ThreadId id) const
    {
        CELER_EXPECT(ptr);
        CELER_EXPECT(id.get() == 0);
        return *ptr;
    }

    explicit operator bool() const { return static_cast<bool>(ptr); }
};

//---------------------------------------------------------------------------//
/*!
 * Delete a VecGeom pool.
 *
 * Due to vecgeom macros, the definition of this function can only be compiled
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
 * Construction of the navstatepool has to be in a host compliation unit due to
 * VecGeom macro magic.
 */
template<>
struct VGNavCollection<Ownership::value, MemSpace::device>
{
    using UPNavStatePool
        = std::unique_ptr<vecgeom::cxx::NavStatePool, NavStatePoolDeleter>;

    //// DATA ////

    UPNavStatePool pool;
    void*          ptr = nullptr;

    //// METHODS ////

    void resize(int max_depth, int size);

    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(ptr);
    }
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Reference on-device memory owned by VGNavCollection<value, device>.
 *
 * The NavStatePool underpinning the storage returns a void pointer that must
 * be manually manipulated to get a single state pointer.
 */
template<>
struct VGNavCollection<Ownership::reference, MemSpace::device>
{
    using NavState = vecgeom::cuda::NavigationState;
    void* ptr      = nullptr;

    VGNavCollection& operator=(const VGNavCollection&) = default;

    void operator=(VGNavCollection<Ownership::value, MemSpace::device>& other);

    CELER_FUNCTION NavState& at(int max_depth, ThreadId thread) const
    {
        CELER_EXPECT(ptr);
        CELER_EXPECT(max_depth > 0);
#    ifdef __NVCC__
        // This code only compiles when run through CUDA so it must be
        // escaped.
        char* result = reinterpret_cast<char*>(this->ptr);
        result += vecgeom::cuda::NavigationState::SizeOfInstanceAlignAware(
                      max_depth)
                  * thread.get();
        return *reinterpret_cast<NavState*>(ptr);
#    else
        (void)sizeof(thread);
        CELER_ASSERT_UNREACHABLE();
#    endif
    }

    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(ptr);
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
