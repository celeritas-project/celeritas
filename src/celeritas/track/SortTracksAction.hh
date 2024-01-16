//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SortTracksAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sort tracks according to a given strategy specified by TrackOrder.
 *
 * This action can be applied at different stage of a simulation step,
 * automatically determined by TrackOrder. This should not have any impact on
 * simulation output: it is only useful for accelerator optimizations.
 */
class SortTracksAction final : public ExplicitActionInterface,
                               public BeginRunActionInterface
{
  public:
    // Construct with action ID and sort criteria
    SortTracksAction(ActionId id, TrackOrder track_order);

    //! Default destructor
    ~SortTracksAction() = default;

    //! Execute the action with host data
    void execute(CoreParams const& params, CoreStateHost& state) const final;

    //! Execute the action with device data
    void execute(CoreParams const& params, CoreStateDevice& state) const final;

    //! Set host data at the beginning of a run
    void begin_run(CoreParams const&, CoreStateHost&) final;

    //! Set device data at the beginning of a run
    void begin_run(CoreParams const&, CoreStateDevice&) final;

    //! ID of the action
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final;

    //! Description of the action for user interaction
    std::string description() const final { return "sort tracks states"; }

    //! Dependency ordering of the action
    ActionOrder order() const final { return action_order_; }

  private:
    ActionId id_;
    ActionOrder action_order_{ActionOrder::size_};
    TrackOrder track_order_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
