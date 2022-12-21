//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveHit.hh
//---------------------------------------------------------------------------//
#pragma once

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
    G4double      edep{0};      //!< energy deposition
    G4double      time{0};      //!< time (global coordinate)
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

    // Accessors
    inline HitData data() const { return data_; }

    // Add energy deposition
    inline void add_edep(G4double edep) { data_.edep += edep; }

  private:
    HitData data_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
