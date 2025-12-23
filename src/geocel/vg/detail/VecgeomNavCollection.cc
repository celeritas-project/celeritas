//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomNavCollection.cc
//---------------------------------------------------------------------------//
#include "VecgeomNavCollection.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HOST VALUE
//---------------------------------------------------------------------------//
/*!
 * Resize with a number of states.
 *
 * This is here only for legacy backward compatibility, so we can justify using
 * a bogus max depth that's too large.
 */
void resize(VecgeomNavCollection<Ownership::value, MemSpace::host>* nav,
            size_type size)
{
    using NavState = vecgeom::NavStatePath;
    auto depth = vecgeom::GeoManager::Instance().getMaxDepth();

    // Add navigation states to collection
    nav->nav_state.resize(size);
    for (UPVgPathState& state : nav->nav_state)
    {
        state = std::unique_ptr<NavState>(NavState::MakeInstance(depth));
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
}  // namespace detail
}  // namespace celeritas
