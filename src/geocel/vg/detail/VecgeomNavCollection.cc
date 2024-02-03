//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomNavCollection.cc
//---------------------------------------------------------------------------//
#include "VecgeomNavCollection.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/navigation/NavStatePool.h>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/Device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HOST VALUE
//---------------------------------------------------------------------------//
/*!
 * Resize with a number of states.
 */
void VecgeomNavCollection<Ownership::value, MemSpace::host>::resize(
    int max_depth, size_type size)
{
    CELER_EXPECT(max_depth > 0);

    // Add navigation states to collection
    this->nav_state.resize(size);
    for (UPNavState& state : this->nav_state)
    {
        state = std::unique_ptr<NavState>(NavState::MakeInstance(max_depth));
    }
}

//---------------------------------------------------------------------------//
// HOST REFERENCE
//---------------------------------------------------------------------------//
/*!
 * Get a reference to host value data.
 */
auto VecgeomNavCollection<Ownership::reference, MemSpace::host>::operator=(
    VecgeomNavCollection<Ownership::value, MemSpace::host>& other)
    -> VecgeomNavCollection&
{
    nav_state = make_span(other.nav_state);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the navigation state at the given thread.
 */
auto VecgeomNavCollection<Ownership::reference, MemSpace::host>::at(
    int, TrackSlotId id) const -> NavState&
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id < nav_state.size());
    return *nav_state[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
// DEVICE VALUE
//---------------------------------------------------------------------------//
/*!
 * Deleter frees CUDA data.
 */
void NavStatePoolDeleter::operator()(arg_type ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Allocate the pool and save the GPU pointer.
 */
void VecgeomNavCollection<Ownership::value, MemSpace::device>::resize(
    int md, size_type sz)
{
    CELER_EXPECT(md > 0);
    CELER_EXPECT(sz > 0);
    CELER_EXPECT(celeritas::device());

    pool.reset(new vecgeom::cxx::NavStatePool(sz, md));
    this->ptr = pool->GetGPUPointer();
    this->max_depth = md;
    this->size = sz;
}

//---------------------------------------------------------------------------//
// DEVICE REFERENCE
//---------------------------------------------------------------------------//
/*!
 * Copy the GPU pointer from the host-managed pool.
 */
auto VecgeomNavCollection<Ownership::reference, MemSpace::device>::operator=(
    VecgeomNavCollection<Ownership::value, MemSpace::device>& other)
    -> VecgeomNavCollection&
{
    CELER_ASSERT(other);
    pool_view = vecgeom::NavStatePoolView{
        (char*)other.ptr, other.max_depth, (int)other.size};
    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
