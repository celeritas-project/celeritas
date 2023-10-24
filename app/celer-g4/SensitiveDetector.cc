//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

#include <memory>
#include <G4HCofThisEvent.hh>
#include <G4SDManager.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "HitRootIO.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(name), hcid_{-1}, collection_{nullptr}
{
    this->collectionName.insert(name);
    HitRootIO::Instance()->AddSensitiveDetector(name);
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
bool SensitiveDetector::ProcessHits(G4Step* g4step, G4TouchableHistory*)
{
    CELER_ASSERT(g4step);
    auto const edep = g4step->GetTotalEnergyDeposit();  // [MeV]

    if (edep == 0)
    {
        return false;
    }
    auto* pre_step = g4step->GetPreStepPoint();
    CELER_ASSERT(pre_step);
    auto* post_step = g4step->GetPostStepPoint();  // Can be undefined
    auto* touchable = g4step->GetPreStepPoint()->GetTouchable();
    CELER_ASSERT(touchable);

    // Insert hit
    HitData hit;
    {
        hit.id = touchable->GetVolume()->GetCopyNo();
        hit.edep = edep;
        hit.time = g4step->GetPreStepPoint()->GetGlobalTime();  // [ns]
    }

    // Insert pre- and post-step data
    StepData step;
    {
        step.energy_loss = edep;  // [MeV]
        step.length = g4step->GetStepLength();  // [mm]

        // Pre- and post-step data
        this->store_step_point(*pre_step, StepData::pre, step);
        if (post_step)
        {
            // Why there's never a defined post-step?
            this->store_step_point(*post_step, StepData::post, step);
        }
    }
    collection_->insert(new SensitiveHit(hit, step));
    return true;
}

//---------------------------------------------------------------------------//
void SensitiveDetector::store_step_point(G4StepPoint& step_point,
                                         StepData::StepType step_type,
                                         StepData& step)
{
    auto const phys_vol = step_point.GetPhysicalVolume();
    CELER_ASSERT(phys_vol);
    auto const log_vol = phys_vol->GetLogicalVolume();
    CELER_ASSERT(log_vol);

    step.detector_id[step_type] = log_vol->GetInstanceID();
    step.energy[step_type] = step_point.GetKineticEnergy();  // [MeV]
    step.time[step_type] = step_point.GetGlobalTime();  // [ns]
    auto const& pos = step_point.GetPosition();
    step.pos[step_type][0] = pos.x();  // [mm]
    step.pos[step_type][1] = pos.y();  // [mm]
    step.pos[step_type][2] = pos.z();  // [mm]
    auto const& dir = step_point.GetMomentumDirection();
    step.dir[step_type][0] = dir.x();
    step.dir[step_type][1] = dir.y();
    step.dir[step_type][2] = dir.z();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
