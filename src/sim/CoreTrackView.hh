//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CoreTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/StackAllocator.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/em/AtomicRelaxationHelper.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/RngEngine.hh"
#include "sim/SimTrackView.hh"

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
    using ParamsRef
        = CoreParamsData<Ownership::const_reference, MemSpace::native>;
    using StateRef = CoreStateData<Ownership::reference, MemSpace::native>;
    //!@}

  public:
    //!@{
    //! Type aliases
    using SecondaryAllocator = StackAllocator<Secondary>;
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

    // Return an RNG engine
    inline CELER_FUNCTION RngEngine make_rng_engine() const;

    // Return a secondary stack allocator
    inline CELER_FUNCTION SecondaryAllocator make_secondary_allocator() const;

    // TODO: don't expose relaxation helper to everyone
    inline CELER_FUNCTION AtomicRelaxationHelper
    make_relaxation_helper(ElementId el_id) const;

    //! Get the track's index among the states
    CELER_FUNCTION ThreadId thread_id() const { return thread_; }

    //!@{
    //! TO BE DELETED: raw access to extra state data
    CELER_FUNCTION real_type& energy_deposition() const
    {
        return states_.energy_deposition[thread_];
    }
    CELER_FUNCTION real_type& step_length() const
    {
        return states_.step_length[thread_];
    }
    CELER_FUNCTION Interaction& interaction() const
    {
        return states_.interactions[thread_];
    }
    CELER_FUNCTION void reset_model_id() const
    {
        // TODO: reset action ID (part of sim, not phys)
        celeritas::PhysicsTrackView phys(
            params_.physics, states_.physics, {}, {}, thread_);
        phys.model_id({});
    }
    //!@}

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
 * Return the RNG engine.
 */
CELER_FUNCTION auto CoreTrackView::make_rng_engine() const -> RngEngine
{
    return RngEngine{states_.rng, thread_};
}

//---------------------------------------------------------------------------//
/*!
 * Return a secondary stack allocator view.
 */
CELER_FUNCTION auto CoreTrackView::make_secondary_allocator() const
    -> SecondaryAllocator
{
    return SecondaryAllocator{states_.secondaries};
}

//---------------------------------------------------------------------------//
/*!
 * Return a secondary stack allocator view.
 */
CELER_FUNCTION auto CoreTrackView::make_relaxation_helper(ElementId el_id) const
    -> AtomicRelaxationHelper
{
    CELER_ASSERT(el_id);
    return AtomicRelaxationHelper{
        params_.relaxation, states_.relaxation, el_id, thread_};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
