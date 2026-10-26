//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.hh
//! \sa DetectorAction.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "corecel/cont/Span.hh"
#include "corecel/data/AsyncCompactedCopy.hh"
#include "corecel/data/AuxInterface.hh"
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
 * an invalid detector ID. On the host, the invalid hits are pruned in place,
 * and a span of only valid hits is then passed into the user provided
 * callback function.
 *
 * On device, the valid hits are compacted into device scratch space and only
 * those are copied to pinned host memory on the memcpy stream, using
 * \c AsyncCompactedCopy . Since the optical transport loop waits for each
 * step before starting the next, the hits of a step are copied while the next
 * step runs and processed during the step after. The remaining hits are
 * processed by \c end_transport , so the callback is called for every step,
 * in order, before transport returns.
 */
class DetectorAction final : public OpticalStepActionInterface,
                             public OpticalEndTransportActionInterface,
                             public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using SPAuxParamsRegistry = std::shared_ptr<AuxParamsRegistry>;
    using CallbackFunc = inp::OpticalDetector::HitCallbackFunc;
    using Hits = AsyncCompactedCopy<DetectorHit>;
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

    //! Host hits are processed during the step
    void end_transport(CoreParams const&, CoreStateHost&) const final {}

    // Process the remaining hits copied from device
    void end_transport(CoreParams const&, CoreStateDevice&) const final;

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

    // Process the hits of the previous steps and copy the hits of the last
    Hits& copy_hits_async(CoreStateDevice&) const;

    // Process all hits remaining from device
    void flush_hits(Hits&) const;

    // Send hits to the callback
    void callback_hits(Span<DetectorHit const>) const;
};

//---------------------------------------------------------------------------//
/*!
 * Persistent per-stream storage for compacted detector hits on device.
 *
 * The device scratch and pinned host buffers are allocated and sized once in
 * \c DetectorAction::create_state so that no reallocation occurs during
 * stepping.
 *
 * This is \em only done when running on device; in host memory, we operate
 * directly on the hit vector.
 */
struct DetectorActionState : public AuxStateInterface
{
    //! Compaction and copy of device hits
    std::optional<AsyncCompactedCopy<DetectorHit>> hits;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
