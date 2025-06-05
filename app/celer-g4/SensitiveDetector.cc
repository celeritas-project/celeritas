//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "geocel/g4/Convert.hh"

#include "GlobalSetup.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with sensitive detector name.
 */
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(name)
{
    this->collectionName.insert(name);
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
 *
 * \todo This should be rewritten/discarded/replaced with EDM4HEP or something
 * else useful.
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
    // NOTE: instance ID will differ from volume ID if geometry has been
    // reloaded
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
