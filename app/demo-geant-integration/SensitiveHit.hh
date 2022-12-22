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
    using PHitAllocator = G4Allocator<SensitiveHit>*;

  public:
    explicit SensitiveHit(const HitData& data);

    //! Accessor the hit data
    const HitData& data() const { return data_; }

    // Overload of operator new
    inline void* operator new(size_t);

  private:
    HitData                            data_;
    static G4ThreadLocal PHitAllocator allocator_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Overload the operator new with G4Allocator.
 */
inline void* SensitiveHit::operator new(size_t)
{
    if (!allocator_)
    {
        allocator_ = new G4Allocator<SensitiveHit>;
    }
    return (void*)allocator_->MallocSingle();
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
