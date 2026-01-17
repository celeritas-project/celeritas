//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize track states.
 *
 * Tracks created from secondaries produced in this action will have the
 * geometry state copied over from the parent instead of initialized from the
 * position. If there are more empty slots than new secondaries, they will be
 * filled by any track initializers remaining from previous steps using the
 * position.
 */
class InitializeTracksAction final : public CoreStepActionInterface
{
  public:
    //! Construct with explicit Id
    explicit InitializeTracksAction(ActionId id) : id_(id) {}

    //! Execute the action with host data
    void step(CoreParams const& params, CoreStateHost& state) const final;

    //! Execute the action with device data
    void step(CoreParams const& params, CoreStateDevice& state) const final;

    //! ID of the action
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final { return "initialize-tracks"; }

    //! Description of the action for user interaction
    std::string_view description() const final
    {
        return "initialize track states";
    }

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::start; }

  private:
    ActionId id_;

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void step_impl(CoreParams const&, CoreStateHost&, size_type) const;
    void step_impl(CoreParams const&, CoreStateDevice&, size_type) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
