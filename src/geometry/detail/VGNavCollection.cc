           //----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavCollection.cc
//---------------------------------------------------------------------------//
#include "VGNavCollection.hh"

#include <VecGeom/navigation/NavStatePool.h>
#include "comm/Device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HOST VALUE
//---------------------------------------------------------------------------//
void VGNavCollection<Ownership::value, MemSpace::host>::resize(int max_depth,
                                                               size_type size)
{
    CELER_EXPECT(max_depth > 0);
    CELER_EXPECT(size == 1);

    nav_state.reset(NavState::MakeInstance(max_depth));
}

//---------------------------------------------------------------------------//
// HOST REFERENCE
//---------------------------------------------------------------------------//
/*!
 * Get a reference to host value data.
 */
void VGNavCollection<Ownership::reference, MemSpace::host>::operator=(
    VGNavCollection<Ownership::value, MemSpace::host>& other)
{
    ptr = other.nav_state.get();
}

//---------------------------------------------------------------------------//
/*!
 * Get the navigation state at the given thread, which must be zero.
 */
auto VGNavCollection<Ownership::reference, MemSpace::host>::at(int,
                                                               ThreadId id) const
    -> NavState&
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id.get() == 0);
    return *ptr;
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
void VGNavCollection<Ownership::value, MemSpace::device>::resize(int       md,
                                                                 size_type sz)
{
    CELER_EXPECT(md > 0);
    CELER_EXPECT(sz > 0);
    CELER_EXPECT(celeritas::device());

    pool.reset(new vecgeom::cxx::NavStatePool(sz, md));
    this->ptr       = pool->GetGPUPointer();
    this->max_depth = md;
    this->size      = sz;
}

//---------------------------------------------------------------------------//
// DEVICE REFERENCE
//---------------------------------------------------------------------------//
/*!
 * Copy the GPU pointer from the host-managed pool.
 */
void VGNavCollection<Ownership::reference, MemSpace::device>::operator=(
    VGNavCollection<Ownership::value, MemSpace::device>& other)
{
    CELER_ASSERT(other);
    ptr       = other.ptr;
    max_depth = other.max_depth;
    size      = other.size;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
