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

#include "GlobalSetup.hh"
#include "RootIO.hh"

namespace celeritas
{
namespace app
{

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Helper function for returning an std::array from a G4ThreeVector.
 */
std::array<double, 3>
make_array(G4ThreeVector const& vec, double const unit = 1)
{
    return {vec.x() / unit, vec.y() / unit, vec.z() / unit};
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with sensitive detector name.
 */
SensitiveDetector::SensitiveDetector(std::string name)
    : G4VSensitiveDetector(name), hcid_{-1}, collection_{nullptr}
{
    this->collectionName.insert(name);

    if (GlobalSetup::Instance()->GetWriteSDHits())
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
    auto const* phys_vol = pre_step->GetPhysicalVolume();
    CELER_ASSERT(phys_vol);
    auto const* log_vol = phys_vol->GetLogicalVolume();
    CELER_ASSERT(log_vol);

    // Insert step data
    EventStepData step;
    {
        EventHitData hit;
        {
            hit.volume = log_vol->GetInstanceID();
            hit.copy_num = phys_vol->GetCopyNo();
            hit.energy_dep = edep / CLHEP::MeV;
            hit.time = pre_step->GetGlobalTime() / CLHEP::s;
        }
        step.hit = std::move(hit);
        step.length = g4step->GetStepLength() / CLHEP::cm;

        // Pre- and post-step data
        step.step_points[static_cast<int>(StepPoint::pre)]
            = this->make_step_point(*pre_step);
        if (auto* post_step = g4step->GetPostStepPoint())
        {
            step.step_points[static_cast<int>(StepPoint::post)]
                = this->make_step_point(*post_step);
        }
    }

    collection_->insert(new SensitiveHit(step));
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Assign pre- and post-step information.
 */
EventStepPointData SensitiveDetector::make_step_point(G4StepPoint& g4step_point)
{
    auto const* phys_vol = g4step_point.GetPhysicalVolume();
    CELER_ASSERT(phys_vol);
    auto const* log_vol = phys_vol->GetLogicalVolume();
    CELER_ASSERT(log_vol);

    EventStepPointData step_point;
    step_point.energy = g4step_point.GetKineticEnergy() / CLHEP::MeV;
    step_point.pos = make_array(g4step_point.GetPosition(), CLHEP::cm);
    step_point.dir = make_array(g4step_point.GetMomentumDirection());
    return step_point;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
