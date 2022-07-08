//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

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
    //! Type aliases
    using ParamsRef = ::celeritas::NativeCRef<CoreParamsData>;
    using StateRef  = ::celeritas::NativeRef<CoreStateData>;
    //!@}

  public:
    // Construct with comprehensive param/state data and thread
    inline CELER_FUNCTION CoreTrackView(const ParamsRef& params,
                                        const StateRef&  states,
                                        ThreadId         thread);

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

    // Return a cutoff view
    inline CELER_FUNCTION CutoffView make_cutoff_view() const;

    // Return a physics view
    inline CELER_FUNCTION PhysicsTrackView make_physics_view() const;

    // Return a view to temporary physics data
    inline CELER_FUNCTION PhysicsStepView make_physics_step_view() const;

    // Return an RNG engine
    inline CELER_FUNCTION RngEngine make_rng_engine() const;

    //! Get the track's index among the states
    CELER_FUNCTION ThreadId thread_id() const { return thread_; }

    // Action ID for encountering a geometry boundary
    inline CELER_FUNCTION ActionId boundary_action() const;

  private:
    const StateRef&  states_;
    const ParamsRef& params_;
    const ThreadId   thread_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with comprehensive param/state data and thread.
 */
CELER_FUNCTION
CoreTrackView::CoreTrackView(const ParamsRef& params,
                             const StateRef&  states,
                             ThreadId         thread)
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
    return SimTrackView{states_.sim, thread_};
}

//---------------------------------------------------------------------------//
/*!
 * Return a geometry view.
 */
CELER_FUNCTION auto CoreTrackView::make_geo_view() const -> GeoTrackView
{
    return GeoTrackView{params_.geometry, states_.geometry, thread_};
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
    return MaterialTrackView{params_.materials, states_.materials, thread_};
}

//---------------------------------------------------------------------------//
/*!
 * Return a particle view.
 */
CELER_FUNCTION auto CoreTrackView::make_particle_view() const
    -> ParticleTrackView
{
    return ParticleTrackView{params_.particles, states_.particles, thread_};
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
        params_.physics, states_.physics, par_id, mat_id, thread_};
}

//---------------------------------------------------------------------------//
/*!
 * Return a physics view.
 */
CELER_FUNCTION auto CoreTrackView::make_physics_step_view() const
    -> PhysicsStepView
{
    return PhysicsStepView{params_.physics, states_.physics, thread_};
}

//---------------------------------------------------------------------------//
/*!
 * Return the RNG engine.
 */
CELER_FUNCTION auto CoreTrackView::make_rng_engine() const -> RngEngine
{
    return RngEngine{states_.rng, thread_};
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
} // namespace celeritas
