//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/track/TrackInitUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize track states.
 *
 * Provides Action interface to `celeritas::initialize_tracks`.
 *
 * \sa celeritas::initialize_tracks
 */
class InitializeTracksAction final : public ExplicitActionInterface
{
  public:
    //! Construct with explicit Id
    explicit InitializeTracksAction(ActionId id) : id_(id) {}

    //! Default destructor
    ~InitializeTracksAction() = default;

    //! Execute the action with host data
    void execute(CoreHostRef const& core) const final
    {
        initialize_tracks(core);
    }

    //! Execute the action with device data
    void execute(CoreDeviceRef const& core) const final;

    //! ID of the action
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final { return "initialize-tracks"; }

    //! Description of the action for user interaction
    std::string description() const final { return "initialize track states"; }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::start; }

  private:
    ActionId id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

inline void
InitializeTracksAction::execute([[maybe_unused]] CoreDeviceRef const& core) const
{
#if !CELER_USE_DEVICE
    CELER_NOT_CONFIGURED("CUDA OR HIP");
#else
    initialize_tracks(core);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
