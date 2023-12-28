//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeoNavCollection.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"

template<class>
class G4ReferenceCountedHandle;
class G4VPhysicalVolume;
class G4TouchableHistory;
class G4Navigator;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Collection-like container for managing Geant4 navigation states.
 *
 * The default is a null-op implementation. Overloads for the host memory space
 * are defined below.
 */
template<Ownership W, MemSpace M>
struct GeantGeoNavCollection
{
    explicit CELER_FUNCTION operator bool() const { return false; }
    CELER_FUNCTION TrackSlotId::size_type size() const { return 0; }
    template<Ownership W2, MemSpace M2>
    CELER_FUNCTION GeantGeoNavCollection&
    operator=(GeantGeoNavCollection<W2, M2>&)
    {
        return *this;
    }
};

//---------------------------------------------------------------------------//

template<class T>
struct G4ExternDeleter
{
    void operator()(T* ptr) noexcept;
};

//---------------------------------------------------------------------------//

using TouchHandle = G4ReferenceCountedHandle<G4TouchableHistory>;
using UPTouchHandle
    = std::unique_ptr<TouchHandle, G4ExternDeleter<TouchHandle>>;
using UPNavigator = std::unique_ptr<G4Navigator, G4ExternDeleter<G4Navigator>>;

//---------------------------------------------------------------------------//
// HOST MEMSPACE
//---------------------------------------------------------------------------//
/*!
 * Manage navigation states in host memory.
 */
template<>
struct GeantGeoNavCollection<Ownership::value, MemSpace::host>
{
    std::vector<UPTouchHandle> touch_handles;
    std::vector<UPNavigator> navigators;

    // Resize with a number of states on the given Geant4 thread ID
    void resize(size_type size, G4VPhysicalVolume* world, StreamId sid);

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return touch_handles.size();
    }

    //! True if constructed properly
    explicit operator bool() const
    {
        return !touch_handles.empty()
               && navigators.size() == touch_handles.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Reference a host-owned navigation state.
 */
template<>
struct GeantGeoNavCollection<Ownership::reference, MemSpace::host>
{
    Span<UPTouchHandle> touch_handles;
    Span<UPNavigator> navigators;

    // Obtain reference from host memory
    GeantGeoNavCollection&
    operator=(GeantGeoNavCollection<Ownership::value, MemSpace::host>& other);
    // Default assignment
    GeantGeoNavCollection& operator=(GeantGeoNavCollection const&) = default;

    // Get the navigation state for a given track slot
    TouchHandle& touch_handle(TrackSlotId tid) const;
    // Get the navigation state for a given track slot
    G4Navigator& navigator(TrackSlotId tid) const;

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return touch_handles.size();
    }

    //! True if constructed properly
    explicit operator bool() const
    {
        return !touch_handles.empty()
               && navigators.size() == touch_handles.size();
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
