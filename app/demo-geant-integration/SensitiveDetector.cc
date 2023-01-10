//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

#include <memory>
#include <G4HCofThisEvent.hh>
#include <G4SDManager.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4String.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include "corecel/Assert.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(name), hcid_{-1}, collection_{nullptr}
{
    this->collectionName.insert(name + "_HC");
}

//---------------------------------------------------------------------------//
/*!
 * Set up hit collections for a new event.
 */
void SensitiveDetector::Initialize(G4HCofThisEvent* hce)
{
    auto collection = std::make_unique<SensitiveHitsCollection>(
        this->SensitiveDetectorName, this->collectionName[0]);

    if (hcid_ < 0)
    {
        // Initialize during the first event. The SD collection was registered
        // inside DetectorConstruction::ConstructSDandField on the local
        // thread.
        hcid_ = G4SDManager::GetSDMpointer()->GetCollectionID(collection.get());
        CELER_ASSERT(hcid_ >= 0);
    }

    // Save a pointer to the collection we just made before tranferring
    // ownership to the HC manager for the event.
    collection_ = collection.get();
    hce->AddHitsCollection(hcid_, collection.release());
    CELER_ENSURE(collection_);
}

//---------------------------------------------------------------------------//
/*!
 * Add hits to the current hit collection.
 */
bool SensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    auto edep = step->GetTotalEnergyDeposit();

    if (edep == 0)
    {
        return false;
    }

    // Create a hit for this step
    auto* touchable = step->GetPreStepPoint()->GetTouchable();
    CELER_ASSERT(touchable);

    unsigned int id = touchable->GetVolume()->GetCopyNo();
    HitData data{id,
                 edep,
                 step->GetPreStepPoint()->GetGlobalTime(),
                 touchable->GetTranslation()};

    collection_->insert(new SensitiveHit(data));

    return true;
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
