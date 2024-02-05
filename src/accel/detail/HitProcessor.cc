//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include <G4Track.hh>
#include <G4TransportationManager.hh>
#include <G4VSensitiveDetector.hh>
#include <G4Version.hh>

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "geocel/g4/Convert.geant.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "TouchableUpdater.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(SPConstVecLV detector_volumes,
                           VecParticle const& particles,
                           StepSelection const& selection,
                           bool locate_touchable)
    : detector_volumes_(std::move(detector_volumes))
{
    CELER_EXPECT(detector_volumes_ && !detector_volumes_->empty());
    CELER_VALIDATE(!locate_touchable || selection.points[StepPoint::pre].pos,
                   << "cannot set 'locate_touchable' because the pre-step "
                      "position is not being collected");
    CELER_VALIDATE(!locate_touchable || selection.points[StepPoint::pre].pos,
                   << "cannot set 'locate_touchable' because the pre-step "
                      "position is not being collected");

    // Create step and step-owned structures
    step_ = std::make_unique<G4Step>();
    step_->NewSecondaryVector();

#if G4VERSION_NUMBER >= 1103
#    define HP_CLEAR_STEP_POINT(CMD) step_->CMD(nullptr)
#else
#    define HP_CLEAR_STEP_POINT(CMD) /* no "reset" before v11.0.3 */
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
    if (locate_touchable)
    {
        CELER_ASSERT(selection.points[StepPoint::pre].pos
                     && selection.points[StepPoint::pre].dir);

        // Create navigator
        G4VPhysicalVolume* world_volume
            = G4TransportationManager::GetTransportationManager()
                  ->GetNavigatorForTracking()
                  ->GetWorldVolume();
        navi_ = std::make_unique<G4Navigator>();
        navi_->SetWorldVolume(world_volume);

        touch_handle_ = new G4TouchableHistory;
        step_->GetPreStepPoint()->SetTouchableHandle(touch_handle_);
    }

    // Create track if user requested particle types
    for (G4ParticleDefinition const* pd : particles)
    {
        CELER_ASSERT(pd);
        tracks_.emplace_back(new G4Track(
            new G4DynamicParticle(pd, G4ThreeVector()), 0.0, G4ThreeVector()));
        tracks_.back()->SetTrackID(-1);
        tracks_.back()->SetParentID(-1);
    }

    // Create secondary vector if using track data

    // Convert logical volumes (global) to sensitive detectors (thread local)
    CELER_LOG_LOCAL(debug) << "Setting up " << detector_volumes_->size()
                           << " sensitive detectors";
    detectors_.resize(detector_volumes_->size());
    for (auto i : range(detectors_.size()))
    {
        G4LogicalVolume const* lv = (*detector_volumes_)[i];
        CELER_ASSERT(lv);
        detectors_[i] = lv->GetSensitiveDetector();
        CELER_VALIDATE(detectors_[i],
                       << "no sensitive detector is attached to volume '"
                       << lv->GetName() << "'@"
                       << static_cast<void const*>(lv));
    }

    CELER_ENSURE(!detectors_.empty());
}

//---------------------------------------------------------------------------//
HitProcessor::~HitProcessor() = default;

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void HitProcessor::operator()(StepStateHostRef const& states)
{
    copy_steps(&steps_, states);
    if (steps_)
    {
        (*this)(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void HitProcessor::operator()(StepStateDeviceRef const& states)
{
    copy_steps(&steps_, states);
    if (steps_)
    {
        (*this)(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate and call hits from a detector output.
 *
 * In an application setting, this is always called with our local data \c
 * steps_ as an argument. For tests, we can call this function explicitly using
 * local test data.
 */
void HitProcessor::operator()(DetectorStepOutput const& out) const
{
    CELER_EXPECT(!out.detector.empty());
    CELER_ASSERT(!navi_ || !out.points[StepPoint::pre].pos.empty());
    CELER_ASSERT(!navi_ || !out.points[StepPoint::pre].dir.empty());
    CELER_ASSERT(tracks_.empty() || !out.particle.empty());

    CELER_LOG_LOCAL(debug) << "Processing " << out.size() << " hits";

    EnumArray<StepPoint, G4StepPoint*> points
        = {step_->GetPreStepPoint(), step_->GetPostStepPoint()};

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

        for (auto sp : range(StepPoint::size_))
        {
            if (!points[sp])
            {
                continue;
            }
            HP_SET(points[sp]->SetGlobalTime, out.points[sp].time, clhep_time);
            HP_SET(points[sp]->SetPosition, out.points[sp].pos, clhep_length);
            HP_SET(points[sp]->SetKineticEnergy,
                   out.points[sp].energy,
                   CLHEP::MeV);
            HP_SET(points[sp]->SetMomentumDirection, out.points[sp].dir, 1);
        }
#undef HP_SET

        if (navi_)
        {
            G4LogicalVolume const* lv = this->detector_volume(out.detector[i]);

            // Update navigation state
            constexpr auto sp = StepPoint::pre;
            TouchableUpdater update_touchable{navi_.get(), touch_handle_()};

            bool success = update_touchable(
                out.points[sp].pos[i], out.points[sp].dir[i], lv);
            if (CELER_UNLIKELY(!success))
            {
                // Inconsistent touchable: skip this energy deposition
                CELER_LOG_LOCAL(error)
                    << "Omitting energy deposition of "
                    << step_->GetTotalEnergyDeposit() / CLHEP::MeV << " [MeV]";
                continue;
            }

            // Copy attributes from logical volume
            points[sp]->SetMaterial(lv->GetMaterial());
            points[sp]->SetMaterialCutsCouple(lv->GetMaterialCutsCouple());
            points[sp]->SetSensitiveDetector(lv->GetSensitiveDetector());
        }

        if (!tracks_.empty())
        {
            this->update_track(out.particle[i]);
        }

        // Hit sensitive detector
        this->detector(out.detector[i])->Hit(step_.get());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Recreate the track from the particle ID and saved post-step data.
 *
 * This is a bit like \c G4Step::UpdateTrack .
 */
void HitProcessor::update_track(ParticleId id) const
{
    CELER_EXPECT(id < tracks_.size());
    G4Track& track = *tracks_[id.unchecked_get()];
    step_->SetTrack(&track);

    if (G4StepPoint* pre_step = step_->GetPreStepPoint())
    {
        // Copy data from track to pre-step
        G4ParticleDefinition const& pd = *track.GetParticleDefinition();
        pre_step->SetCharge(pd.GetPDGCharge());
    }
    if (G4StepPoint* post_step = step_->GetPostStepPoint())
    {
        // Copy data from post-step to track
        track.SetGlobalTime(post_step->GetGlobalTime());
        track.SetPosition(post_step->GetPosition());
        track.SetKineticEnergy(post_step->GetKineticEnergy());
        track.SetMomentumDirection(post_step->GetMomentumDirection());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
