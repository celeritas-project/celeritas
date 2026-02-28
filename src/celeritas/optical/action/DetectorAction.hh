//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "celeritas/inp/Scoring.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/DetectorData.hh"

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Record sensitive detector data for optical photons at the end of every step.
 *
 * The \c DetectorExecutor is responsible for copying hit data for every photon
 * into the state buffer at the end of every step on a kernel level. Even if a
 * track was not in a detector, it is still copied into the state buffer with
 * an invalid detector ID. All hits are copied into pinned memory on the host,
 * where invalid hits are erased. A span of only valid hits is then passed into
 * the user provided callback function.
 */
class DetectorAction final : public OpticalStepActionInterface,
                             public StaticConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using CallbackFunc = typename inp::OpticalDetector::HitCallbackFunc;
    //!@}

  public:
    // Construct with ID and callback function
    explicit DetectorAction(ActionId, CallbackFunc const&);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    //// TYPES ////

    using VecHit = std::vector<DetectorHit>;

    //// DATA ////

    CallbackFunc callback_;

    //// HELPER FUNCTIONS ////

    // Copy hits from device
    VecHit load_hits_sync(CoreStateDevice const&) const;

    // Send hits to the callback
    void callback_hits(VecHit const&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
