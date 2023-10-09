//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/SensitiveHit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4Allocator.hh>
#include <G4ThreeVector.hh>
#include <G4VHit.hh>

#include "HitData.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Example sensitive hit class.
 */
class SensitiveHit final : public G4VHit
{
  public:
    // Construct with hit data
    explicit SensitiveHit(HitData const& data);

    //! Accessor
    HitData const& data() const { return data_; }

    // Overload new/delete to use a custom allocator.
    inline void* operator new(std::size_t);
    inline void operator delete(void*);

  private:
    //// DATA ////

    HitData data_;

    //// HELPER FUNCTIONS ////

    using HitAllocator = G4Allocator<SensitiveHit>;
    static HitAllocator& allocator();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Use G4Allocator to allocate memory for a SensitiveHit.
 */
inline void* SensitiveHit::operator new(std::size_t)
{
    return SensitiveHit::allocator().MallocSingle();
}

//---------------------------------------------------------------------------//
/*!
 * Use G4Allocator to release memory for a SensitiveHit.
 */
inline void SensitiveHit::operator delete(void* hit)
{
    SensitiveHit::allocator().FreeSingle(static_cast<SensitiveHit*>(hit));
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
