//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/StepDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/user/StepDiagnosticBase.hh"

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Tally number of steps taken by each photon in the optical stepping loop.
 */
class StepDiagnostic final : public StepDiagnosticBase,
                             public OpticalStepActionInterface
{
  public:
    // Construct and add to core params
    static std::shared_ptr<StepDiagnostic>
    make_and_insert(CoreParams const& core, size_type max_bins);

    //! Construct with ID and counts
    StepDiagnostic(ActionId id, size_type max_bins, size_type num_streams);

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string_view label() const final { return "optical-step-diagnostic"; }
    // Description of the action for user interaction
    std::string_view description() const final;
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    //!@}

  private:
    ActionId id_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
