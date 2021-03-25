//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavCollection.cc
//---------------------------------------------------------------------------//
#include "VGNavCollection.hh"

#include <VecGeom/navigation/NavStatePool.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HOST VALUE
//---------------------------------------------------------------------------//
void VGNavCollection<Ownership::value, MemSpace::host>::resize(int max_depth,
                                                               int size)
{
    CELER_EXPECT(max_depth > 0);
    CELER_EXPECT(size == 1);

    nav_state.reset(NavState::MakeInstance(max_depth));
}

//---------------------------------------------------------------------------//
// HOST REFERENCE
//---------------------------------------------------------------------------//
void VGNavCollection<Ownership::reference, MemSpace::host>::operator=(
    VGNavCollection<Ownership::value, MemSpace::host>& other)
{
    ptr = other.nav_state.get();
}

//---------------------------------------------------------------------------//
// DEVICE VALUE
//---------------------------------------------------------------------------//
void VGNavCollection<Ownership::value, MemSpace::device>::resize(int max_depth,
                                                                 int size)
{
    CELER_EXPECT(max_depth > 0);
    CELER_EXPECT(size > 0);

    pool.reset(new vecgeom::cxx::NavStatePool(size, max_depth));
    ptr = pool->GetGPUPointer();
}

//---------------------------------------------------------------------------//
//! Deleter frees cuda data
void NavStatePoolDeleter::operator()(arg_type ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
// DEVICE REFERENCE
//---------------------------------------------------------------------------//
void VGNavCollection<Ownership::reference, MemSpace::device>::operator=(
    VGNavCollection<Ownership::value, MemSpace::device>& other)
{
    CELER_ASSERT(other);
    ptr = other.ptr;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
