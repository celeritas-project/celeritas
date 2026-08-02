//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.hh
//! \sa DetectorAction.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "celeritas/inp/Scoring.hh"
#include "celeritas/optical/DetectorData.hh"

#include "ActionInterface.hh"

namespace celeritas
{
class ActionRegistry;
class AuxParamsRegistry;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Record sensitive detector data for optical photons at the end of every step.
 *
 * The \c DetectorExecutor is responsible for copying hit data for every photon
 * into the state buffer at the end of every step on a kernel level. Even if a
 * track was not in a detector, it is still copied into the state buffer with
 * an invalid detector ID. All hits are copied into a persistent pinned-memory
 * buffer (allocated once as auxiliary state, sized to the number of track
 * slots), where invalid hits are pruned by slicing a span over the valid
 * prefix. A span of only valid hits is then passed into the user provided
 * callback function.
 */
class DetectorAction final : public OpticalStepActionInterface,
                             public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using SPAuxParamsRegistry = std::shared_ptr<AuxParamsRegistry>;
    using CallbackFunc = inp::OpticalDetector::HitCallbackFunc;
    //!@}

  public:
    // Create and add to core params
    static std::shared_ptr<DetectorAction> make_and_insert(
        SPActionRegistry const&,
        SPAuxParamsRegistry const&,
        CallbackFunc const&);

    // Construct with action ID, aux ID, and callback function
    DetectorAction(ActionId, AuxId, CallbackFunc const&);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }

    //!@{
    //! \name Action interface

    //! ID of the action
    ActionId action_id() const final { return sad_.action_id(); }
    //! Short name for the action (also satisfies AuxParamsInterface::label)
    std::string_view label() const final { return sad_.label(); }
    //! Description of the action
    std::string_view description() const final { return sad_.description(); }
    //!@}

    //!@{
    //! \name Aux interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build auxiliary state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

  private:
    //// DATA ////

    StaticActionData sad_;
    AuxId aux_id_;
    CallbackFunc callback_;

    //// HELPER FUNCTIONS ////

    // Copy hits from device
    Span<DetectorHit const> load_hits_sync(CoreStateDevice const&) const;

    // Send hits to the callback
    void callback_hits(Span<DetectorHit const>) const;
};

//---------------------------------------------------------------------------//
/*!
 * Persistent per-stream storage for pruned detector hits on device.
 *
 * The pinned buffer is allocated and sized once in \c
 * DetectorAction::create_state so that no reallocation occurs during
 * stepping.
 *
 * This is \em only done when running on device; in host memory, we operate
 * directly on the hit vector.
 */
struct DetectorActionState : public AuxStateInterface
{
    using VecHit = std::vector<DetectorHit, PinnedAllocator<DetectorHit>>;

    VecHit hits;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
