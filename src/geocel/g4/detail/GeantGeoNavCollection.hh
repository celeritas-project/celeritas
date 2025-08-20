//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/detail/GeantGeoNavCollection.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/GeantGeoUtils.hh"

template<class>
class G4ReferenceCountedHandle;
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
    CELER_FUNCTION void reset() { CELER_ASSERT_UNREACHABLE(); }
};

//---------------------------------------------------------------------------//

template<class T>
struct G4ExternDeleter
{
    void operator()(T* ptr) noexcept;
};

template<class T>
using GeantUniquePtr = std::unique_ptr<T, G4ExternDeleter<T>>;

//---------------------------------------------------------------------------//

using GeantTouchableHandle = G4ReferenceCountedHandle<GeantTouchableBase>;
using UPTouchHandle = GeantUniquePtr<GeantTouchableHandle>;
using UPNavigator = GeantUniquePtr<G4Navigator>;

//---------------------------------------------------------------------------//
// HOST MEMSPACE
//---------------------------------------------------------------------------//
/*!
 * Manage navigation states in host memory.
 *
 * Each NavCollection should be used on a single CPU thread.
 * Geant4 typically is built in multithreaded mode, in which case lots of data
 * under the hood is "thread local", preventing the use of Geant4
 * geometry states in an OpenMP context.
 *
 * \todo I think we can use collections for the handles+navigators.
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

    // Clean up on the original thread, necessary for thread-local G4 alloc
    void reset() { CELER_ASSERT_UNREACHABLE(); }
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

    // Default constructors
    GeantGeoNavCollection() = default;
    GeantGeoNavCollection(GeantGeoNavCollection const&) = default;

    // Obtain reference from host memory
    GeantGeoNavCollection&
    operator=(GeantGeoNavCollection<Ownership::value, MemSpace::host>& other);
    // Default assignment
    GeantGeoNavCollection& operator=(GeantGeoNavCollection const&) = default;

    // Get the navigation state for a given track slot
    GeantTouchableHandle& touch_handle(TrackSlotId tid) const;
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
               && navigators.size() == touch_handles.size()
               && touch_handles.front() && navigators.front();
    }

    // Clean up on the original thread, necessary for thread-local G4 alloc
    void reset();
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
