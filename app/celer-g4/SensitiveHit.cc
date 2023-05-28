//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveHit.cc
//---------------------------------------------------------------------------//
#include "SensitiveHit.hh"

#include <G4Types.hh>

#include "corecel/Macros.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct and access a thread-local allocator.
 *
 * The allocator must never be deleted because destroying it seems to corrupt
 * the application's memory.
 */
auto SensitiveHit::allocator() -> HitAllocator&
{
    static G4ThreadLocal HitAllocator* alloc_;
    if (CELER_UNLIKELY(!alloc_))
    {
        alloc_ = new HitAllocator;
    }
    return *alloc_;
}

//---------------------------------------------------------------------------//
/*!
 * Constructor.
 */
SensitiveHit::SensitiveHit(HitData const& data) : G4VHit(), data_{data} {}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
