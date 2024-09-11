//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ThreadId.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/track/SimTrackView.hh"

// #include "MaterialTrackView.hh"
#include "PhysicsStepView.hh"
#include "PhysicsTrackView.hh"
#include "TrackData.hh"
#include "TrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class CoreTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<CoreParamsData>;
    using StateRef = NativeRef<CoreStateData>;
    //!@}

  public:
    // Construct with core param/state data and thread
    inline CELER_FUNCTION CoreTrackView(ParamsRef const& params,
                                        StateRef const& states,
                                        ThreadId thread);

    // Construct directly from a track slot ID
    inline CELER_FUNCTION CoreTrackView(ParamsRef const& params,
                                        StateRef const& states,
                                        TrackSlotId slot);

    //// VIEW CONSTRUCTORS ////

    inline CELER_FUNCTION SimTrackView make_sim_view() const;
    inline CELER_FUNCTION GeoTrackView make_geo_view() const;
    // inline CELER_FUNCTION MaterialTrackView make_material_view() const;
    inline CELER_FUNCTION TrackView make_particle_view() const;
    inline CELER_FUNCTION PhysicsTrackView make_physics_view() const;
    inline CELER_FUNCTION PhysicsStepView make_physics_step_view() const;
    inline CELER_FUNCTION RngEngine make_rng_engine() const;

    //// ACCESSORS ////

    CELER_FORCEINLINE_FUNCTION ThreadId thread_id() const;
    CELER_FORCEINLINE_FUNCTION TrackSlotId track_slot_id() const;

    inline CELER_FUNCTION CoreScalars const& core_scalars() const;

  private:
    StateRef const& states_;
    ParamsRef const& params_;
    ThreadId const thread_id_;
    TrackSlotId track_slot_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION CoreTrackView::CoreTrackView(ParamsRef const& params,
                                            StateRef const& states,
                                            ThreadId thread)
    : states_(states), params_(params), thread_id_(thread)
{
    // TODO: no track slots?
    track_slot_id_ = TrackSlotId{thread_id_.get()};

    CELER_ENSURE(track_slot_id_ < states_.size());
}

CELER_FUNCTION CoreTrackView::CoreTrackView(ParamsRef const& params,
                                            StateRef const& states,
                                            TrackSlotId slot)
    : states_(states), params_(params), track_slot_id_(slot)
{
    CELER_EXPECT(track_slot_id_ < states_.size());
}

CELER_FUNCTION SimTrackView CoreTrackView::make_sim_view() const
{
    return SimTrackView{params_.sim, states_.sim, this->track_slot_id()};
}

CELER_FUNCTION GeoTrackView CoreTrackView::make_geo_view() const
{
    return GeoTrackView{
        params_.geometry, states_.geometry, this->track_slot_id()};
}

// CELER_FUNCTION MaterialTrackView CoreTrackView::make_material_view() const
// {
//     return MaterialTrackView{params_.materials, states_.materials,
//     this->track_slot_id()};
// }

CELER_FUNCTION TrackView CoreTrackView::make_particle_view() const
{
    // TODO: IMPLEMENT ME
    return TrackView{units::MevEnergy{0.0136}, {1, 0, 0}};
}

CELER_FUNCTION PhysicsTrackView CoreTrackView::make_physics_view() const
{
    // TODO: replace with MaterialTrackView to get ID
    OpticalMaterialId opt_mat_id = states_.materials[this->track_slot_id()];
    CELER_ASSERT(opt_mat_id);
    return PhysicsTrackView{
        params_.physics, states_.physics, opt_mat_id, this->track_slot_id()};
}

CELER_FUNCTION PhysicsStepView CoreTrackView::make_physics_step_view() const
{
    return PhysicsStepView{
        params_.physics, states_.physics, this->track_slot_id()};
}

CELER_FUNCTION auto CoreTrackView::make_rng_engine() const -> RngEngine
{
    return RngEngine{params_.rng, states_.rng, this->track_slot_id()};
}

CELER_FORCEINLINE_FUNCTION ThreadId CoreTrackView::thread_id() const
{
    CELER_EXPECT(thread_id_);
    return thread_id_;
}

CELER_FORCEINLINE_FUNCTION TrackSlotId CoreTrackView::track_slot_id() const
{
    return track_slot_id_;
}

CELER_FUNCTION CoreScalars const& CoreTrackView::core_scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
