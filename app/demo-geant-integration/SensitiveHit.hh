//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveHit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4Allocator.hh>
#include <G4ThreeVector.hh>
#include <G4VHit.hh>

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Example sensitive hit data.
 */
struct HitData
{
    unsigned int  id{0};        //!< detector id
    double        edep{0};      //!< energy deposition
    double        time{0};      //!< time (global coordinate)
    G4ThreeVector pos{0, 0, 0}; //!< position (global coordinate)
};

//---------------------------------------------------------------------------//
/*!
 * Example sensitive hit class.
 */
class SensitiveHit final : public G4VHit
{
  public:
    explicit SensitiveHit(const HitData& data);

    //! Accessor the hit data
    const HitData& data() const { return data_; }

    // Overload new/delete to use a custom allocator.
    inline void* operator new(std::size_t);
    inline void  operator delete(void*);

  private:
    using HitAllocator = G4Allocator<SensitiveHit>;

    HitData data_;

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
} // namespace demo_geant
