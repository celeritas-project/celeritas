//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.cc
//---------------------------------------------------------------------------//
#include "HitProcessor.hh"

#include <string>
#include <utility>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4TransportationManager.hh>
#include <G4VSensitiveDetector.hh>
#include <G4Version.hh>

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TraceCounter.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "LevelTouchableUpdater.hh"
#include "NaviTouchableUpdater.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(SPConstVecLV detector_volumes,
                           SPConstGeo const& geo,
                           VecParticle const& particles,
                           StepSelection const& selection,
                           bool locate_touchable)
    : detector_volumes_(std::move(detector_volumes))
{
    CELER_EXPECT(detector_volumes_ && !detector_volumes_->empty());
    CELER_EXPECT(geo);

    CELER_LOG_LOCAL(debug) << "Setting up hit processor for "
                           << detector_volumes_->size()
                           << " sensitive detectors";

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
        // Create touchable updater
        touch_handle_ = new G4TouchableHistory;
        step_->GetPreStepPoint()->SetTouchableHandle(touch_handle_);
        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // ORANGE doesn't yet support level reconstruction: see
            // HitManager.cc
            CELER_EXPECT(selection.points[StepPoint::pre].pos
                         && selection.points[StepPoint::pre].dir);
            update_touchable_
                = std::make_unique<NaviTouchableUpdater>(detector_volumes_);
        }
        else
        {
            CELER_EXPECT(selection.points[StepPoint::pre].volume_instance_ids);
            update_touchable_ = std::make_unique<LevelTouchableUpdater>(geo);
        }
    }

    // Set invalid values for unsupported SD attributes
    step_->SetNonIonizingEnergyDeposit(
        -std::numeric_limits<double>::infinity());
    for (G4StepPoint* p : {step_->GetPreStepPoint(), step_->GetPostStepPoint()})
    {
        if (!p)
        {
            continue;
        }
        // Time since track was created
        p->SetLocalTime(std::numeric_limits<double>::infinity());
        // Time in rest frame since track was created
        p->SetProperTime(std::numeric_limits<double>::infinity());
        // Speed
        p->SetVelocity(std::numeric_limits<double>::infinity());
        // Safety distance
        p->SetSafety(std::numeric_limits<double>::infinity());
        // Polarization (default to zero)
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

    // Convert logical volumes (global) to sensitive detectors (thread local)
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
    CELER_ENSURE(static_cast<bool>(update_touchable_) == locate_touchable);
}

//---------------------------------------------------------------------------//
//! Log on destruction
HitProcessor::~HitProcessor()
{
    try
    {
        CELER_LOG_LOCAL(debug) << "Deallocating hit processor";
    }
    catch (...)  // NOLINT(bugprone-empty-catch)
    {
        // Ignore anything bad that happens while logging
    }
}

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
    CELER_ASSERT(tracks_.empty() || !out.particle.empty());

    ScopedProfiling profile_this{"process-hits"};
    trace_counter("process-hits", out.size());

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

        G4LogicalVolume const* lv = this->detector_volume(out.detector[i]);

        HP_SET(step_->SetTotalEnergyDeposit, out.energy_deposition, CLHEP::MeV);
        HP_SET(step_->SetStepLength, out.step_length, clhep_length);

        if (update_touchable_)
        {
            // Update navigation state
            bool success = (*update_touchable_)(out, i, touch_handle_());

            if (CELER_UNLIKELY(!success))
            {
                // Inconsistent touchable: skip this energy deposition
                CELER_LOG_LOCAL(error)
                    << "Omitting energy deposition of "
                    << step_->GetTotalEnergyDeposit() / CLHEP::MeV << " [MeV]";
                continue;
            }
        }

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
            /*!
             * \todo Celeritas currently ignores incoming particle weight and
             * does not perform any variance reduction. See issue #1268.
             */
            points[sp]->SetWeight(1.0);
        }
#undef HP_SET

        // Copy attributes from logical volume
        points[StepPoint::pre]->SetMaterial(lv->GetMaterial());
        points[StepPoint::pre]->SetMaterialCutsCouple(
            lv->GetMaterialCutsCouple());
        points[StepPoint::pre]->SetSensitiveDetector(
            lv->GetSensitiveDetector());

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

    G4ParticleDefinition const& pd = *track.GetParticleDefinition();

    for (G4StepPoint* p : {step_->GetPreStepPoint(), step_->GetPostStepPoint()})
    {
        if (!p)
        {
            continue;
        }

        // Copy data from track to step points
        p->SetMass(pd.GetPDGMass());
        p->SetCharge(pd.GetPDGCharge());
    }
    if (G4StepPoint* post_step = step_->GetPostStepPoint())
    {
        // Copy data from post-step to track
        track.SetGlobalTime(post_step->GetGlobalTime());
        track.SetPosition(post_step->GetPosition());
        track.SetKineticEnergy(post_step->GetKineticEnergy());
        track.SetMomentumDirection(post_step->GetMomentumDirection());
        track.SetWeight(post_step->GetWeight());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
