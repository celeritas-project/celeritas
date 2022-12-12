//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

#include <G4SystemOfUnits.hh>

#include "corecel/io/Logger.hh"
namespace demo_geant
{
//---------------------------------------------------------------------------//
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(std::move(name))
{
    CELER_LOG_LOCAL(debug) << "Constructing SensitiveDetector '"
                           << this->GetName() << "'";
}

//---------------------------------------------------------------------------//
void SensitiveDetector::Initialize(G4HCofThisEvent*)
{
    CELER_LOG_LOCAL(debug) << "SensitiveDetector::Initialize ("
                           << this->GetName() << ")";
}

//---------------------------------------------------------------------------//
G4bool SensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    auto edep = step->GetTotalEnergyDeposit();

    if (edep == 0)
    {
        return false;
    }

    auto time = step->GetPreStepPoint()->GetGlobalTime();
    CELER_LOG_LOCAL(debug) << "Deposited " << edep / CLHEP::MeV << " MeV into "
                           << this->GetName() << " at " << time;

    return true;
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
