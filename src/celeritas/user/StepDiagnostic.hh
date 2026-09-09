//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "StepDiagnosticBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Tally number of steps taken by each particle type in the core stepping loop.
 */
class StepDiagnostic final : public StepDiagnosticBase,
                             public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<StepDiagnostic> make_and_insert(
        CoreParams const& core, size_type max_bins);

    //! Construct with particle data
    StepDiagnostic(ActionId id,
                   SPConstParticle particle,
                   size_type max_bins,
                   size_type num_streams);

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string_view label() const final { return "step-diagnostic"; }
    // Description of the action for user interaction
    std::string_view description() const final;
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    //!@}

  private:
    ActionId id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
