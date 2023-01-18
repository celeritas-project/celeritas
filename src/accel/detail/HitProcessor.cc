//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.cc
//---------------------------------------------------------------------------//
#include "HitProcessor.hh"

#include <string>
#include <utility>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>
#include <G4Navigator.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistory.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSensitiveDetector.hh>
#include <G4Version.hh>

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
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
inline T convert_to_geant(T const& val, T units)
{
    return val * units;
}

//---------------------------------------------------------------------------//
inline G4ThreeVector convert_to_geant(Real3 const& arr, double units)
{
    return {arr[0] * units, arr[1] * units, arr[2] * units};
}

//---------------------------------------------------------------------------//
inline double convert_to_geant(units::MevEnergy const& energy, double units)
{
    CELER_EXPECT(units == CLHEP::MeV);
    return energy.value() * CLHEP::MeV;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(VecLV detector_volumes,
                           StepSelection const& selection,
                           bool locate_touchable)
    : detector_volumes_(std::move(detector_volumes))
{
    CELER_EXPECT(!detector_volumes_.empty());
    CELER_VALIDATE(!locate_touchable || selection.points[StepPoint::pre].pos,
                   << "cannot set 'locate_touchable' because the pre-step "
                      "position is not being collected");

    // Create temporary objects
    step_ = std::make_unique<G4Step>();

#if G4VERSION_NUMBER >= 1101
#    define HP_CLEAR_STEP_POINT(CMD) step_->CMD(nullptr)
#else
#    define HP_CLEAR_STEP_POINT(CMD) /* no "reset" before v11.0.1 */
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
#undef HP_CLEAR_STEP_POINT

    // Create navigator
    G4VPhysicalVolume* world_volume
        = G4TransportationManager::GetTransportationManager()
              ->GetNavigatorForTracking()
              ->GetWorldVolume();
    if (locate_touchable && selection.points[StepPoint::pre].pos)
    {
        navi_ = std::make_unique<G4Navigator>();
        navi_->SetWorldVolume(world_volume);

        // Create "touchable handle" (shared pointer to G4TouchableHistory)
        touch_handle_ = new G4TouchableHistory;
        step_->GetPreStepPoint()->SetTouchableHandle(touch_handle_);
    }
}

//---------------------------------------------------------------------------//
HitProcessor::~HitProcessor() = default;

//---------------------------------------------------------------------------//
/*!
 * Generate and call hits from a detector output.
 */
void HitProcessor::operator()(DetectorStepOutput const& out) const
{
    CELER_EXPECT(!out.detector.empty());

    CELER_LOG_LOCAL(debug) << "Processing " << out.size() << " hits";

    for (auto i : range(out.size()))
    {
#define HP_SET(SETTER, OUT, UNITS)                   \
    do                                               \
    {                                                \
        if (!OUT.empty())                            \
        {                                            \
            SETTER(convert_to_geant(OUT[i], UNITS)); \
        }                                            \
    } while (0)

        HP_SET(step_->SetTotalEnergyDeposit, out.energy_deposition, CLHEP::MeV);
        // TODO: how to handle these attributes?
        // step_->SetTrack(primary_track);

        // TODO: assert that event ID is consistent with active
        // LocalTransporter event?

        EnumArray<StepPoint, G4StepPoint*> points
            = {step_->GetPreStepPoint(), step_->GetPostStepPoint()};
        for (auto sp : range(StepPoint::size_))
        {
            if (!points[sp])
            {
                continue;
            }
            HP_SET(points[sp]->SetGlobalTime, out.points[sp].time, CLHEP::s);
            HP_SET(points[sp]->SetPosition, out.points[sp].pos, CLHEP::cm);
            HP_SET(points[sp]->SetKineticEnergy,
                   out.points[sp].energy,
                   CLHEP::MeV);
            // TODO: do we set secondary properties like the velocity,
            // material, mass, charge,  ... ?

            // TODO: how to handle these attributes?
            // step_->SetTrack(primary_track);
            // dynamic particle, ParticleDefinition
            // pre->SetLocalTime
            // pre->SetProperTime
            // pre->SetTouchableHandle
        }
#undef HP_SET

        if (navi_)
        {
            // Locate pre-step point
            navi_->LocateGlobalPointAndUpdateTouchable(
                points[StepPoint::pre]->GetPosition(),
                touch_handle_(),
                /* relative_search = */ false);
            // TODO: can we be sure we're in the right volume on the first step
            // inside?
        }

        // Hit sensitive detector (NOTE: GetSensitiveDetector returns a
        // thread-local object from a global object.)
        CELER_ASSERT(out.detector[i] < detector_volumes_.size());
        G4VSensitiveDetector* sd
            = detector_volumes_[out.detector[i].unchecked_get()]
                  ->GetSensitiveDetector();
        CELER_ASSERT(sd);
        sd->Hit(step_.get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
