//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveHit.cc
//---------------------------------------------------------------------------//
#include "SensitiveHit.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct and access a thread-local allocator.
 */
auto SensitiveHit::allocator() -> HitAllocator&
{
    static G4ThreadLocal HitAllocator alloc_;
    return alloc_;
}

//---------------------------------------------------------------------------//
/*!
 * Constructor.
 */
SensitiveHit::SensitiveHit(const HitData& data) : G4VHit(), data_{data} {}

//---------------------------------------------------------------------------//
} // namespace demo_geant
