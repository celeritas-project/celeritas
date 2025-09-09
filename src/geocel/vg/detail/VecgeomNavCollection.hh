//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomNavCollection.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/navigation/NavStateFwd.h>
#include <VecGeom/navigation/NavStatePool.h>
#include <VecGeom/navigation/NavigationState.h>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"

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
struct VecgeomNavCollection;

//---------------------------------------------------------------------------//
// HOST MEMSPACE
//---------------------------------------------------------------------------//
/*!
 * Manage navigation states in host memory.
 */
template<>
struct VecgeomNavCollection<Ownership::value, MemSpace::host>
{
    using NavState = vecgeom::cxx::NavigationState;
    using UPNavState = std::unique_ptr<NavState>;

    std::vector<UPNavState> nav_state;

    // Resize with a number of states
    void resize(int max_depth, size_type size);
    // Whether the collection is assigned
    explicit operator bool() const { return !nav_state.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Reference a host-owned navigation state.
 */
template<>
struct VecgeomNavCollection<Ownership::reference, MemSpace::host>
{
    using NavState = vecgeom::cxx::NavigationState;
    using UPNavState = std::unique_ptr<NavState>;

    Span<UPNavState> nav_state;

    // Default construction and copy construction
    VecgeomNavCollection() = default;
    VecgeomNavCollection(VecgeomNavCollection const&) = default;

    // Obtain reference from host memory
    VecgeomNavCollection&
    operator=(VecgeomNavCollection<Ownership::value, MemSpace::host>& other);
    // Default copy assignment
    VecgeomNavCollection& operator=(VecgeomNavCollection const&) = default;

    // Get the navigation state for a given track slot
    NavState& at(int, TrackSlotId tid) const;
    //! True if the collection is assigned/valiid
    explicit operator bool() const { return !nav_state.empty(); }
};

//---------------------------------------------------------------------------//
// DEVICE MEMSPACE
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
struct VecgeomNavCollection<Ownership::value, MemSpace::device>
{
    using UPNavStatePool
        = std::unique_ptr<vecgeom::cxx::NavStatePool, NavStatePoolDeleter>;

    UPNavStatePool pool;
    void* ptr = nullptr;
    int max_depth = 0;
    size_type size = 0;

    // Resize based on geometry params and state size
    void resize(int max_depth, size_type size);
    //! True if the collection is assigned/valid
    explicit CELER_FUNCTION operator bool() const { return ptr; }
};

//---------------------------------------------------------------------------//
/*!
 * Reference on-device memory owned by VecgeomNavCollection<value, device>.
 *
 * The NavStatePool underpinning the storage returns a void pointer that must
 * be manually manipulated to get a single state pointer. The max_depth
 * argument must be the same as the given to VecgeomGeoParams.
 */
template<>
struct VecgeomNavCollection<Ownership::reference, MemSpace::device>
{
    using NavState = vecgeom::NavigationState;

    vecgeom::NavStatePoolView pool_view = {nullptr, 0, 0};

    // Default construct and copy construct
    VecgeomNavCollection() = default;
    VecgeomNavCollection(VecgeomNavCollection const& other) = default;

    // Assign from device value
    VecgeomNavCollection&
    operator=(VecgeomNavCollection<Ownership::value, MemSpace::device>& other);
    // Assign from device reference
    VecgeomNavCollection& operator=(VecgeomNavCollection const& other)
        = default;

    // Get the navigation state for the given track slot
    inline CELER_FUNCTION NavState& at(int max_depth, TrackSlotId tid) const;

    //! True if the collection is assigned/valid
    explicit CELER_FUNCTION operator bool() const
    {
        return pool_view.IsValid();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Get the navigation state at the given track slot.
 *
 * The max_depth_param is used for error checking against the allocated
 * max_depth.
 */
CELER_FUNCTION auto
VecgeomNavCollection<Ownership::reference, MemSpace::device>::at(
    int max_depth_param, TrackSlotId tid) const -> NavState&
{
    CELER_EXPECT(this->pool_view.IsValid());
    CELER_EXPECT(tid < this->pool_view.Capacity());
    CELER_EXPECT(max_depth_param == this->pool_view.Depth());

    return *const_cast<NavState*>((this->pool_view)[tid.get()]);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
