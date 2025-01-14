//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/SensitiveHit.cc
//---------------------------------------------------------------------------//
#include "SensitiveHit.hh"

#include <G4Types.hh>

#include "corecel/Macros.hh"

namespace celeritas
{
namespace app
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
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static G4ThreadLocal HitAllocator* alloc_;
    if (CELER_UNLIKELY(!alloc_))
    {
        alloc_ = new HitAllocator;
    }
    return *alloc_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with hit data.
 */
SensitiveHit::SensitiveHit(EventHitData const& hit) : G4VHit(), data_{hit} {}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
