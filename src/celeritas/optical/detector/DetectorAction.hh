//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionInterface.hh"

#include "DetectorData.hh"
#include "ScoringParams.hh"

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
    // Construct with ID
    explicit DetectorAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    // Process hits copied from the kernels and send them to the callback
    template<MemSpace M>
    void process_hits(CoreParams const&, CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 *
 * Copied hits might be invalid, and are removed before sending into the
 * callback function. The callback is only execute when a non-zero amount of
 * valid hits occurs.
 */
template<MemSpace M>
void DetectorAction::process_hits(CoreParams const& params,
                                  CoreState<M>& state) const
{
    DetectorHitOutput hit_results;

    // Copy hits (possibly from device) into pinned vector
    copy_hits<M>(&hit_results, state.ref().scoring, state.stream_id());

    // Erase all hits with invalid detector ID
    hit_results.hits.erase(
        std::remove_if(hit_results.hits.begin(),
                       hit_results.hits.end(),
                       [](DetectorHit const& hit) {
                           return !static_cast<bool>(hit.detector);
                       }),
        hit_results.hits.end());

    // Send hits to the callback function, if there are any
    if (!hit_results.hits.empty())
    {
        auto scoring = params.scoring();
        CELER_ASSERT(scoring);
        scoring->process_hits(make_span(hit_results.hits));
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
