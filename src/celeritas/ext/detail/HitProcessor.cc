//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HitProcessor.cc
//---------------------------------------------------------------------------//
#include "HitProcessor.hh"

#include <cstddef>
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

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TraceCounter.hh"
#include "geocel/GeantGeoParams.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "LevelTouchableUpdater.hh"
#include "../GeantStepPointView.hh"
#include "../GeantStepView.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get the geometry step status when volume instance IDs are present.
 *
 * Note that this isn't entirely accurate if crossing from one
 * replica/parameterised region to another. For that we will need to map
 * distinct (instance id, replica number) to unique volume instances. Perhaps
 * that would be better done by using "touchables" globally and reconstructing
 * volume instances in post.
 */
G4StepStatus
get_step_status(DetectorStepOutput const& out, size_type step_index)
{
    auto pre = LevelTouchableUpdater::volume_instances(
        out, step_index, StepPoint::pre);
    auto post = LevelTouchableUpdater::volume_instances(
        out, step_index, StepPoint::post);
    for (auto i : range(out.num_volume_levels))
    {
        if (pre[i] != post[i])
        {
            // Volume instance changed
            if (i == 0 && post[i] == VolumeInstanceId{})
            {
                // Exited the geometry
                return G4StepStatus::fWorldBoundary;
            }
            // Changed volumes
            return G4StepStatus::fGeomBoundary;
        }
        if (!pre[i])
        {
            // Empty volume sentinel encountered: exit
            break;
        }
    }
    return G4StepStatus::fUserDefinedLimit;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(SPConstVecLV detector_volumes,
                           VecParticle const& particles,
                           StepSelection const& selection,
                           StepPointBool const& locate_touchable)
    : detector_volumes_(std::move(detector_volumes))
    , step_{std::make_shared<G4Step>()}
    , track_reconstruction_{particles, step_}
    , step_post_status_{
          selection.points[StepPoint::pre].volume_instance_ids
          && selection.points[StepPoint::post].volume_instance_ids}
{
    CELER_EXPECT(detector_volumes_ && !detector_volumes_->empty());

    // Even though this is created locally, all threads should be doing the
    // same thing
    CELER_LOG(debug) << "Setting up thread-local hit processor for "
                     << detector_volumes_->size() << " sensitive detectors";

    // Allocate secondary vector, needed to keep some SDs from crashing
    step_->NewSecondaryVector();

    GeantStepView step_view{*step_};

    for (auto sp : range(StepPoint::size_))
    {
        if (!selection.points[sp])
        {
            step_view.delete_step_point(sp);
            CELER_ASSERT(!locate_touchable[sp]);
        }
        else
        {
            auto point_view = step_view.step_point(sp);
            point_view.clear_unsupported();
            step_points_[sp] = &point_view.step_point();
            if (locate_touchable[sp])
            {
                // Create touchable handle for this step point
                touch_handle_[sp] = new G4TouchableHistory;
                step_points_[sp]->SetTouchableHandle(touch_handle_[sp]);
                if (!update_touchable_)
                {
                    CELER_EXPECT(selection.points[sp].volume_instance_ids);
                    // FIXME: pass geant geo into this constructor
                    auto ggeo = ::celeritas::global_geant_geo().lock();
                    CELER_ASSERT(ggeo);
                    update_touchable_ = std::make_unique<LevelTouchableUpdater>(
                        std::move(ggeo));
                }
            }
        }
    }

    // Set invalid values for unsupported SD attributes
    step_->SetNonIonizingEnergyDeposit(
        -std::numeric_limits<double>::infinity());

    // Convert logical volumes (global) to sensitive detectors (thread local)
    detectors_.resize(detector_volumes_->size());
    for (auto i : range(detectors_.size()))
    {
        G4LogicalVolume const* lv = (*detector_volumes_)[i];
        CELER_ASSERT(lv);
        detectors_[i] = lv->GetSensitiveDetector();
        CELER_VALIDATE(detectors_[i],
                       << "no sensitive detector is attached to volume '"
                       << StreamableLV{lv});
    }

    CELER_ENSURE(!detectors_.empty());
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
        num_hits_ += steps_.size();
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
        num_hits_ += steps_.size();
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
    ScopedProfiling profile_this{"process-hits"};
    trace_counter("process-hits", out.size());
    for (auto i : range(out.size()))
    {
        (*this)(out, i);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate and call a single hit.
 */
void HitProcessor::operator()(DetectorStepOutput const& out, size_type i) const
{
    CELER_EXPECT(!out.detector.empty());
    CELER_EXPECT(i < out.size());

    GeantStepView step_view{*step_};
    if (!out.energy_deposition.empty())
    {
        step_view.energy_deposition(
            GeantStepView::Energy{out.energy_deposition[i].value()});
    }
    if (!out.step_length.empty())
    {
        step_view.step_length(
            native_value_to<GeantStepView::Length>(out.step_length[i]));
    }

    for (auto sp : range(StepPoint::size_))
    {
        auto* g4sp = step_points_[sp];
        if (!g4sp)
        {
            continue;
        }

        if (auto& touch_handle = touch_handle_[sp])
        {
            CELER_ASSERT(update_touchable_);
            // Update navigation state
            bool success = (*update_touchable_)(out, i, sp, touch_handle());

            if (CELER_UNLIKELY(!success))
            {
                // Inconsistent touchable: skip this energy deposition
                CELER_LOG_LOCAL(error) << "Omitting energy deposition of "
                                       << step_view.energy_deposition();
                return;
            }
        }

        auto const& celer_sp = out.points[sp];
        GeantStepPointView sp_view{*g4sp};

        if (!celer_sp.time.empty())
        {
            sp_view.time(
                native_value_to<GeantStepPointView::Time>(celer_sp.time[i]));
        }
        if (!celer_sp.pos.empty())
        {
            sp_view.pos(
                native_value_to<GeantStepPointView::Length>(celer_sp.pos[i]));
        }
        if (!celer_sp.energy.empty())
        {
            sp_view.energy(
                GeantStepPointView::Energy{celer_sp.energy[i].value()});
        }
        if (!celer_sp.dir.empty())
        {
            sp_view.dir(static_array_cast<double>(celer_sp.dir[i]));
        }
        if (!out.weight.empty())
        {
            // Celeritas weight does not currently change across a step
            sp_view.weight(out.weight[i]);
        }

        // Copy attributes from logical volume
        if (sp == StepPoint::pre)
        {
            G4LogicalVolume const* lv = this->detector_volume(out.detector[i]);
            CELER_ASSERT(lv);
            // Use lv already known from the in-volume detector
            sp_view.update_from_volume(*lv);
        }
        else
        {
            // Look up LV from the touchable
            sp_view.update_from_volume();
        }
    }

    if (!out.particle.empty())
    {
        G4Track& g4track = track_reconstruction_.view(
            out.particle[i],
            !out.primary_id.empty() ? out.primary_id[i] : PrimaryId{});
        CELER_ASSERT(&g4track == step_->GetTrack());
        // Copy step information to the corresponding track
        GeantStepView{*step_}.update_track();
    }

    if (step_post_status_)
    {
        // Update the post-step status based on the geometry instances
        auto* g4sp = step_->GetPostStepPoint();
        CELER_ASSERT(g4sp);
        g4sp->SetStepStatus(get_step_status(out, i));
    }

    // Hit sensitive detector
    this->detector(out.detector[i])->Hit(step_.get());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
