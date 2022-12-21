//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4THitsCollection.hh>
#include <G4VSensitiveDetector.hh>

#include "SensitiveHit.hh"

class G4Step;
class G4String;
class G4HCofThisEvent;

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Example sensitive detector.
 */
class SensitiveDetector final : public G4VSensitiveDetector
{
  public:
    //!@{
    //! \name Type aliases
    using SensitiveHitsCollection = G4THitsCollection<SensitiveHit>;

    //!@}

  public:
    explicit SensitiveDetector(G4String name);

  protected:
    void   Initialize(G4HCofThisEvent*) final;
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) final;

  private:
    G4int                                    hcid_;
    std::unique_ptr<SensitiveHitsCollection> collection_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
