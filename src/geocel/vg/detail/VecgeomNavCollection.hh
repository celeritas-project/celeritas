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
#include <VecGeom/navigation/NavStatePath.h>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/vg/VecgeomTypes.hh"

#if CELER_VGNAV != CELER_VGNAV_PATH
#    error \
        "This file can only be included when using VecGeom built with VecGeom_NAV=path"
#endif

namespace celeritas
{
namespace detail
{
using UPVgPathState = std::unique_ptr<vecgeom::NavStatePath>;

//---------------------------------------------------------------------------//
/*!
 * Collection-like container for managing VecGeom navigation states.
 *
 * This is now used explicitly for a host-only collection of "path" navigation
 * states.
 */
template<Ownership W, MemSpace M>
struct VecgeomNavCollection
{
    explicit operator bool() const { return false; }
    size_type size() const { return 0; }

    template<Ownership W2, MemSpace M2>
    VecgeomNavCollection& operator=(VecgeomNavCollection<W2, M2>&)
    {
        CELER_ASSERT_UNREACHABLE();
    }
};

//---------------------------------------------------------------------------//
// HOST MEMSPACE
//---------------------------------------------------------------------------//
/*!
 * Manage navigation states in host memory.
 */
template<>
struct VecgeomNavCollection<Ownership::value, MemSpace::host>
{
    std::vector<UPVgPathState> nav_state;

    // Whether the collection is assigned
    explicit operator bool() const { return !nav_state.empty(); }

    //! Number of elements
    size_type size() const { return nav_state.size(); }
};

//---------------------------------------------------------------------------//
/*!
 * Reference a host-owned navigation state.
 */
template<>
struct VecgeomNavCollection<Ownership::reference, MemSpace::host>
{
    Span<UPVgPathState> nav_state;

    // Default construction and copy construction
    VecgeomNavCollection() = default;
    VecgeomNavCollection(VecgeomNavCollection const&) = default;

    // Obtain reference from host memory
    VecgeomNavCollection&
    operator=(VecgeomNavCollection<Ownership::value, MemSpace::host>& other);
    // Default copy assignment
    VecgeomNavCollection& operator=(VecgeomNavCollection const&) = default;

    //! Get the navigation state for a given track slot
    vecgeom::NavStatePath& operator[](TrackSlotId tid) const
    {
        CELER_EXPECT(*this);
        CELER_EXPECT(tid < nav_state.size());
        return *nav_state[tid.unchecked_get()];
    }

    //! Number of elements
    size_type size() const { return nav_state.size(); }

    //! True if the collection is assigned/valid
    explicit operator bool() const { return !nav_state.empty(); }
};

//---------------------------------------------------------------------------//
// Resize with a number of states
void resize(VecgeomNavCollection<Ownership::value, MemSpace::host>* nav,
            size_type size);

inline void
resize(VecgeomNavCollection<Ownership::value, MemSpace::device>*, size_type)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
