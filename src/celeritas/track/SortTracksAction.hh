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
 * Sort track by status
 *
 * Provides Action interface to
 * `celeritas::detail::partition_tracks_by_status`.
 *
 * TODO: Use init params.init to configure sorting strategy
 *
 * \sa celeritas::detail::partition_tracks_by_status
 */
class SortTracksAction final : public ExplicitActionInterface
{
  public:
    //! Construct with explicit Id
    explicit SortTracksAction(ActionId id,
                              ActionOrder action_order,
                              TrackOrder track_order)
        : id_(id), action_order_(action_order), track_order_(track_order)
    {
    }

    //! Default destructor
    ~SortTracksAction() = default;

    //! Execute the action with host data
    void execute(CoreHostRef const& core) const final;

    //! Execute the action with device data
    void execute(CoreDeviceRef const& core) const final;

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
    ActionOrder action_order_;
    TrackOrder track_order_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
