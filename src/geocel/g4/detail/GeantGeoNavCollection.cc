//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/detail/GeantGeoNavCollection.cc
//---------------------------------------------------------------------------//
#include "GeantGeoNavCollection.hh"

#include <G4Navigator.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "geocel/GeantUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void G4ExternDeleter<T>::operator()(T* ptr) noexcept
{
    delete ptr;
}

template struct G4ExternDeleter<GeantTouchableHandle>;
template struct G4ExternDeleter<G4Navigator>;

//---------------------------------------------------------------------------//
/*!
 * Resize with a number of states.
 *
 * The stream ID is checked against the Geant4 threading because of custom
 * thread-local allocators in Geant4.
 */
void GeantGeoNavCollection<Ownership::value, MemSpace::host>::resize(
    size_type size, G4VPhysicalVolume* world, StreamId sid)
{
    CELER_EXPECT(world);
    CELER_EXPECT(sid.get() == static_cast<size_type>(get_geant_thread_id()));

    // Add navigation states to collection
    this->touch_handles.resize(size);
    this->navigators.resize(size);
    for (size_type i : range(size))
    {
        this->touch_handles[i].reset(new G4TouchableHandle);
        *this->touch_handles[i] = new G4TouchableHistory;
        this->navigators[i].reset(new G4Navigator);
        this->navigators[i]->SetWorldVolume(world);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to host value data.
 */
auto GeantGeoNavCollection<Ownership::reference, MemSpace::host>::operator=(
    GeantGeoNavCollection<Ownership::value, MemSpace::host>& other)
    -> GeantGeoNavCollection&
{
    this->touch_handles = make_span(other.touch_handles);
    this->navigators = make_span(other.navigators);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the touchable handle at the given thread.
 */
auto GeantGeoNavCollection<Ownership::reference, MemSpace::host>::touch_handle(
    TrackSlotId tid) const -> G4TouchableHandle&
{
    CELER_EXPECT(*this);
    CELER_EXPECT(tid < this->size());
    return *this->touch_handles[tid.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the navigation state at the given thread.
 */
auto GeantGeoNavCollection<Ownership::reference, MemSpace::host>::navigator(
    TrackSlotId tid) const -> G4Navigator&
{
    CELER_EXPECT(*this);
    CELER_EXPECT(tid < this->size());
    return *this->navigators[tid.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
