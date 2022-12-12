//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.cc
//---------------------------------------------------------------------------//
#include "HitProcessor.hh"

#include <algorithm>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4TransportationManager.hh>
#include <G4VSensitiveDetector.hh>

#include "corecel/cont/Range.hh"
#include "celeritas/ext/GeantVersion.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
const T& convert_to_geant(const T& val)
{
    return val;
}

//---------------------------------------------------------------------------//
G4ThreeVector convert_to_geant(const Real3& arr)
{
    return {arr[0], arr[1], arr[2]};
}

//---------------------------------------------------------------------------//
double convert_to_geant(const units::MevEnergy& energy)
{
    return energy.value() * CLHEP::MeV;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(const VecLV&         detector_volumes,
                           const StepSelection& selection)
{
    CELER_EXPECT(!detector_volumes.empty());

    // Create temporary objects
    step_ = std::make_unique<G4Step>();

#if CELERITAS_G4_V10
#    define HP_CLEAR_STEP_POINT(CMD) /* no "reset" before v11 */
#else
#    define HP_CLEAR_STEP_POINT(CMD) step_->CMD(nullptr)
#endif

#define HP_SETUP_POINT(LOWER, TITLE)                                          \
    do                                                                        \
    {                                                                         \
        if (!selection.points[StepPoint::LOWER])                              \
        {                                                                     \
            HP_CLEAR_STEP_POINT(Reset##TITLE##StepPoint);                     \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            step_->Get##TITLE##StepPoint()->SetStepStatus(fUserDefinedLimit); \
        }                                                                     \
    } while (0)

    HP_SETUP_POINT(pre, Pre);
    HP_SETUP_POINT(post, Post);
#undef HP_SETUP_POINT

    // Create navigator
    G4VPhysicalVolume* world_volume
        = G4TransportationManager::GetTransportationManager()
              ->GetNavigatorForTracking()
              ->GetWorldVolume();
    navi_ = std::make_unique<G4Navigator>();
    navi_->SetWorldVolume(world_volume);

    // Create "touchable handle" (shared pointer to G4TouchableHistory)
    touch_handle_ = G4TouchableHandle{new G4TouchableHistory};

    // Find sensitive detectors for each detector ID. Note that these are
    // *thread-local* pointers coming from *global* data: see Geant4 "split
    // classes"
    detectors_.reserve(detector_volumes.size());
    for (G4LogicalVolume* lv : detector_volumes)
    {
        CELER_EXPECT(lv);
        detectors_.push_back(lv->GetSensitiveDetector());
        CELER_ENSURE(detectors_.back());
    }

    CELER_ENSURE(detectors_.size() == detector_volumes.size());
}

//---------------------------------------------------------------------------//
/*!
 * Generate and call hits from a detector output.
 */
void HitProcessor::operator()(const DetectorStepOutput& out) const
{
    CELER_EXPECT(!out.detector.empty());

    for (auto i : range(out.size()))
    {
#define HP_SET(SETTER, OUT)                   \
    do                                        \
    {                                         \
        if (!OUT.empty())                     \
        {                                     \
            SETTER(convert_to_geant(OUT[i])); \
        }                                     \
    } while (0)

        HP_SET(step_->SetTotalEnergyDeposit, out.energy_deposition);
        // TODO: how to handle these attributes?
        // step_->SetTrack(primary_track);

        EnumArray<StepPoint, G4StepPoint*> points
            = {step_->GetPreStepPoint(), step_->GetPostStepPoint()};
        for (auto sp : range(StepPoint::size_))
        {
            if (!points[sp])
            {
                continue;
            }
            HP_SET(points[sp]->SetGlobalTime, out.points[sp].time);
            HP_SET(points[sp]->SetPosition, out.points[sp].pos);
            HP_SET(points[sp]->SetMomentumDirection, out.points[sp].dir);

            // TODO: how to handle these attributes?
            // step_->SetTrack(primary_track);
            // pre->SetLocalTime
            // pre->SetProperTime
            // pre->SetTouchableHandle
        }
#undef HP_SET

        // Hit sensitive detector
        CELER_ASSERT(out.detector[i] < detectors_.size());
        detectors_[out.detector[i].unchecked_get()]->Hit(step_.get());
    }
}
#undef HP_SET

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
