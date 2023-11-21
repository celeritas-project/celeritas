//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/SensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4THitsCollection.hh>
#include <G4VSensitiveDetector.hh>

#include "celeritas/Types.hh"

#include "SensitiveHit.hh"

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Example sensitive detector.
 */
class SensitiveDetector final : public G4VSensitiveDetector
{
    //!@{
    //! \name Type aliases
    using SensitiveHitsCollection = G4THitsCollection<SensitiveHit>;
    //!@}

  public:
    explicit SensitiveDetector(std::string name);

  protected:
    void Initialize(G4HCofThisEvent*) final;
    bool ProcessHits(G4Step*, G4TouchableHistory*) final;

  private:
    int hcid_;
    SensitiveHitsCollection* collection_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
