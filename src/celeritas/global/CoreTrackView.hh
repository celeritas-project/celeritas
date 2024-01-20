//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ThreadId.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/track/SimTrackView.hh"

#include "CoreTrackData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to create views from core track data.
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
    // Construct with comprehensive param/state data and thread
    inline CELER_FUNCTION CoreTrackView(ParamsRef const& params,
                                        StateRef const& states,
                                        ThreadId thread);

    // Return a simulation management view
    inline CELER_FUNCTION SimTrackView make_sim_view() const;

    // Return a geometry view
    inline CELER_FUNCTION GeoTrackView make_geo_view() const;

    // Return a geometry-material view
    inline CELER_FUNCTION GeoMaterialView make_geo_material_view() const;

    // Return a material view
    inline CELER_FUNCTION MaterialTrackView make_material_view() const;

    // Return a particle view
    inline CELER_FUNCTION ParticleTrackView make_particle_view() const;

    // Return a particle view of another particle type
    inline CELER_FUNCTION ParticleView make_particle_view(ParticleId) const;

    // Return a cutoff view
    inline CELER_FUNCTION CutoffView make_cutoff_view() const;

    // Return a physics view
    inline CELER_FUNCTION PhysicsTrackView make_physics_view() const;

    // Return a view to temporary physics data
    inline CELER_FUNCTION PhysicsStepView make_physics_step_view() const;

    // Return an RNG engine
    inline CELER_FUNCTION RngEngine make_rng_engine() const;

    //! Get the index of the current thread in the current kernel
    CELER_FUNCTION ThreadId thread_id() const { return thread_; }

    // Get the track's index among the states
    inline CELER_FUNCTION TrackSlotId track_slot_id() const;

    // Action ID for encountering a geometry boundary
    inline CELER_FUNCTION ActionId boundary_action() const;

    // Action ID for some other propagation limit (e.g. field stepping)
    inline CELER_FUNCTION ActionId propagation_limit_action() const;

    // Action ID for being abandoned while looping
    inline CELER_FUNCTION ActionId abandon_looping_action() const;

    // HACK: return scalars (maybe have a struct for all actions?)
    inline CELER_FUNCTION CoreScalars const& core_scalars() const;

  private:
    StateRef const& states_;
    ParamsRef const& params_;
    ThreadId const thread_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with comprehensive param/state data and thread.
 */
CELER_FUNCTION
CoreTrackView::CoreTrackView(ParamsRef const& params,
                             StateRef const& states,
                             ThreadId thread)
    : states_(states), params_(params), thread_(thread)
{
    CELER_EXPECT(thread_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Return a simulation management view.
 */
CELER_FUNCTION SimTrackView CoreTrackView::make_sim_view() const
{
    return SimTrackView{params_.sim, states_.sim, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a geometry view.
 */
CELER_FUNCTION auto CoreTrackView::make_geo_view() const -> GeoTrackView
{
    return GeoTrackView{
        params_.geometry, states_.geometry, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a geometry-material view.
 */
CELER_FUNCTION auto CoreTrackView::make_geo_material_view() const
    -> GeoMaterialView
{
    return GeoMaterialView{params_.geo_mats};
}

//---------------------------------------------------------------------------//
/*!
 * Return a material view.
 */
CELER_FUNCTION auto CoreTrackView::make_material_view() const
    -> MaterialTrackView
{
    return MaterialTrackView{
        params_.materials, states_.materials, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a particle view.
 */
CELER_FUNCTION auto CoreTrackView::make_particle_view() const
    -> ParticleTrackView
{
    return ParticleTrackView{
        params_.particles, states_.particles, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a particle view of another particle type.
 */
CELER_FUNCTION auto CoreTrackView::make_particle_view(ParticleId pid) const
    -> ParticleView
{
    return ParticleView{params_.particles, pid};
}

//---------------------------------------------------------------------------//
/*!
 * Return a cutoff view.
 */
CELER_FUNCTION auto CoreTrackView::make_cutoff_view() const -> CutoffView
{
    MaterialId mat_id = this->make_material_view().material_id();
    CELER_ASSERT(mat_id);
    return CutoffView{params_.cutoffs, mat_id};
}

//---------------------------------------------------------------------------//
/*!
 * Return a physics view.
 */
CELER_FUNCTION auto CoreTrackView::make_physics_view() const -> PhysicsTrackView
{
    MaterialId mat_id = this->make_material_view().material_id();
    CELER_ASSERT(mat_id);
    ParticleId par_id = this->make_particle_view().particle_id();
    CELER_ASSERT(par_id);
    return PhysicsTrackView{
        params_.physics, states_.physics, par_id, mat_id, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a physics view.
 */
CELER_FUNCTION auto CoreTrackView::make_physics_step_view() const
    -> PhysicsStepView
{
    return PhysicsStepView{
        params_.physics, states_.physics, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return the RNG engine.
 */
CELER_FUNCTION auto CoreTrackView::make_rng_engine() const -> RngEngine
{
    return RngEngine{params_.rng, states_.rng, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Get the track's index among the states.
 */
CELER_FUNCTION TrackSlotId CoreTrackView::track_slot_id() const
{
    return TrackSlotId{states_.track_slots[thread_]};
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for encountering a geometry boundary.
 */
CELER_FUNCTION ActionId CoreTrackView::boundary_action() const
{
    return params_.scalars.boundary_action;
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for having to pause the step during propagation.
 *
 * This could be from an internal limiter (number of substeps during field
 * propagation) or from having to "bump" the track position for some reason
 * (geometry issue). The volume *must not* change as a result of the
 * propagation, and this should be an extremely rare case.
 */
CELER_FUNCTION ActionId CoreTrackView::propagation_limit_action() const
{
    return params_.scalars.propagation_limit_action;
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for being abandoned while looping.
 */
CELER_FUNCTION ActionId CoreTrackView::abandon_looping_action() const
{
    return params_.scalars.abandon_looping_action;
}

//---------------------------------------------------------------------------//
/*!
 * Get access to all the core scalars.
 *
 * TODO: maybe have a struct for all actions to simplify the class?
 */
CELER_FUNCTION CoreScalars const& CoreTrackView::core_scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
