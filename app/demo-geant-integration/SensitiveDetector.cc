//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

#include <G4HCofThisEvent.hh>
#include <G4SDManager.hh>
#include <G4Step.hh>
#include <G4String.hh>
#include <G4SystemOfUnits.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(name), hcid_(-1)
{
    this->collectionName.insert(name + "_HC");
}

//---------------------------------------------------------------------------//
/*!
 * Set up hit collections for a new event.
 */
void SensitiveDetector::Initialize(G4HCofThisEvent* hce)
{
    collection_ = std::make_unique<SensitiveHitsCollection>(
        this->SensitiveDetectorName, this->collectionName[0]);
    if (hcid_ < 0)
    {
        // Initialize during the first event
        hcid_
            = G4SDManager::GetSDMpointer()->GetCollectionID(collection_.get());
        CELER_ASSERT(hcid_ >= 0);
    }
    hce->AddHitsCollection(hcid_, collection_.release());
}

//---------------------------------------------------------------------------//
/*!
 * Add hits to the current hit collection
 */
bool SensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    auto edep = step->GetTotalEnergyDeposit();

    if (edep == 0)
    {
        return false;
    }

    // Create a hit for this step
    auto         touchable = step->GetPreStepPoint()->GetTouchable();
    unsigned int id        = touchable->GetVolume()->GetCopyNo();
    HitData      data{id,
                 edep,
                 step->GetPreStepPoint()->GetGlobalTime(),
                 touchable->GetTranslation()};

    collection_->insert(new SensitiveHit(data));

    CELER_LOG_LOCAL(debug) << "Deposited " << edep / CLHEP::MeV << " MeV into "
                           << this->GetName() << " at " << data.time;

    return true;
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
