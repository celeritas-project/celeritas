//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4HCofThisEvent.hh>
#include <G4SDManager.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include "corecel/Assert.hh"
#include "celeritas/ext/Convert.geant.hh"

#include "GlobalSetup.hh"
#include "RootIO.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with sensitive detector name.
 */
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(name), hcid_{-1}, collection_{nullptr}
{
    this->collectionName.insert(name);

    if (RootIO::use_root())
    {
        RootIO::Instance()->AddSensitiveDetector(name);
    }
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

    // Save a pointer to the collection we just made before transferring
    // ownership to the HC manager for the event.
    collection_ = collection.get();
    hce->AddHitsCollection(hcid_, collection.release());
    CELER_ENSURE(collection_);
}

//---------------------------------------------------------------------------//
/*!
 * Add hits to the current hit collection.
 */
bool SensitiveDetector::ProcessHits(G4Step* g4step, G4TouchableHistory*)
{
    CELER_ASSERT(g4step);
    auto const edep = g4step->GetTotalEnergyDeposit();

    if (edep == 0)
    {
        return false;
    }

    auto* pre_step = g4step->GetPreStepPoint();
    CELER_ASSERT(pre_step);
    auto const* phys_vol = pre_step->GetPhysicalVolume();
    CELER_ASSERT(phys_vol);
    auto const* log_vol = phys_vol->GetLogicalVolume();
    CELER_ASSERT(log_vol);

    // Insert hit (use pre-step time since post-steps can be undefined)
    EventHitData hit;
    hit.volume = log_vol->GetInstanceID();
    hit.copy_num = phys_vol->GetCopyNo();
    hit.energy_dep = convert_from_geant(edep, CLHEP::MeV);
    hit.time = convert_from_geant(pre_step->GetGlobalTime(), CLHEP::s);

    collection_->insert(new SensitiveHit(hit));
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
