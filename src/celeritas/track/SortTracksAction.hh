//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
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
 * This action can be applied at different stage of a simulation step, as
 * specified by the ActionOrder. This should not have any impact on simulation
 * output, it is only useful for accelerator optimizations.
 */
class SortTracksAction final : public ExplicitActionInterface
{
  public:
    //! Construct with explicit Id, action order and track order
    explicit SortTracksAction(ActionId id,
                              ActionOrder action_order,
                              TrackOrder track_order)
        : id_(id), action_order_(action_order), track_order_(track_order)
    {
    }

    //! Default destructor
    ~SortTracksAction() = default;

    //! Execute the action with host data
    void execute(CoreParams const& params, StateHostRef& states) const final;

    //! Execute the action with device data
    void execute(CoreParams const& params, StateDeviceRef& states) const final;

    //! ID of the action
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final;

    //! Description of the action for user interaction
    std::string description() const final { return "sort tracks states"; }

    //! Dependency ordering of the action
    ActionOrder order() const final { return action_order_; }

  private:
    template<MemSpace M>
    void execute_impl(CoreParams const& params,
                      CoreStateData<Ownership::reference, M>& states) const;

    ActionId id_;
    ActionOrder action_order_;
    TrackOrder track_order_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
