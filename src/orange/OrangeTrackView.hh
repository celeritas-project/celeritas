//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OrangeTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Macros.hh"
#include "geometry/Types.hh"

#include "Data.hh"
#include "universes/SimpleUnitTracker.hh"
#include "universes/detail/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Operate on the device with shared (persistent) params and local state.
 */
class OrangeTrackView
{
  public:
    //!@{
    //! Type aliases
    using ParamsRef
        = OrangeParamsData<Ownership::const_reference, MemSpace::native>;
    using StateRef = OrangeStateData<Ownership::reference, MemSpace::native>;
    using Initializer_t = GeoTrackInitializer;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        OrangeTrackView& other; //!< Existing geometry
        Real3            dir;   //!< New direction
    };

  public:
    // Construct from params and state params
    inline CELER_FUNCTION OrangeTrackView(const ParamsRef& params,
                                          const StateRef&  states,
                                          ThreadId         tid);

    // Initialize the state
    inline CELER_FUNCTION OrangeTrackView& operator=(const Initializer_t& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION OrangeTrackView&
    operator=(const DetailedInitializer& init);

    //// ACCESSORS ////

    //! The current position
    CELER_FUNCTION const Real3& pos() const { return states_.pos[thread_]; }
    //! The current direction
    CELER_FUNCTION const Real3& dir() const { return states_.dir[thread_]; }
    //! The current volume ID (null if outside)
    CELER_FUNCTION VolumeId volume_id() const { return states_.vol[thread_]; }
    //! The current surface ID
    CELER_FUNCTION SurfaceId surface_id() const
    {
        return states_.surf[thread_];
    }
    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary
    inline CELER_FUNCTION Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the nearest (any direction) boundary within the current volume
    inline CELER_FUNCTION real_type find_safety(const Real3& pos);

    // Move to the boundary in preparation for crossing it
    inline CELER_FUNCTION void move_to_boundary();

    // Move within the volume
    inline CELER_FUNCTION void move_internal(real_type step);

    // Move within the volume to a specific point
    inline CELER_FUNCTION void move_internal(const Real3& pos);

    // Cross from one side of the current surface to the other
    inline CELER_FUNCTION void cross_boundary();

    // Change direction
    inline CELER_FUNCTION void set_dir(const Real3& newdir);

  private:
    //// DATA ////

    const ParamsRef& params_;
    const StateRef&  states_;
    ThreadId         thread_;

    real_type          next_step_{0};   //!< Temporary next step
    detail::OnSurface  next_surface_{}; //!< Temporary next surface

    //// HELPER FUNCTIONS ////

    // Create local sense reference
    inline CELER_FUNCTION Span<Sense> make_temp_sense() const;

    // Create local distance
    inline CELER_FUNCTION detail::TempNextFace make_temp_next() const;

    // Whether the next distance-to-boundary has been found
    CELER_FORCEINLINE_FUNCTION bool has_next_step() const;

    // Invalidate the next distance-to-boundary
    CELER_FORCEINLINE_FUNCTION void clear_next_step();
};

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION
OrangeTrackView::OrangeTrackView(const ParamsRef& params,
                                 const StateRef&  states,
                                 ThreadId         thread)
    : params_(params), states_(states), thread_(thread)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(states_);
    CELER_EXPECT(thread < states.size());

    CELER_ENSURE(!this->has_next_step());
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 *
 * Expensive. This function should only be called to initialize an event from a
 * starting location and direction. Secondaries will initialize their states
 * from a copy of the parent.
 */
CELER_FUNCTION OrangeTrackView&
OrangeTrackView::operator=(const Initializer_t& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    // Save known data to global memory
    states_.pos[thread_]   = init.pos;
    states_.dir[thread_]   = init.dir;
    states_.surf[thread_]  = {};
    states_.sense[thread_] = {};

    // Clear local data
    this->clear_next_step();

    // Create local state
    detail::LocalState local;
    local.pos        = init.pos;
    local.dir        = init.dir;
    local.volume     = {};
    local.surface    = {};
    local.temp_sense = this->make_temp_sense();

    // Initialize logical state
    SimpleUnitTracker tracker(params_);
    auto              tinit = tracker.initialize(local);
    // TODO: error correction/graceful failure if initialiation failured
    CELER_ASSERT(tinit.volume && !tinit.surface);

    // Save local data
    states_.vol[thread_] = tinit.volume;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state from a direction and a copy of the parent state.
 */
CELER_FUNCTION
OrangeTrackView& OrangeTrackView::operator=(const DetailedInitializer& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));
    CELER_EXPECT(states_.vol[init.other.thread_]);

    // Copy init track's position but update the direction
    states_.pos[thread_]   = states_.pos[init.other.thread_];
    states_.dir[thread_]   = init.dir;
    states_.vol[thread_]   = states_.vol[init.other.thread_];
    states_.surf[thread_]  = states_.surf[init.other.thread_];
    states_.sense[thread_] = states_.sense[init.other.thread_];

    // Clear step and surface info
    this->clear_next_step();

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation OrangeTrackView::find_next_step()
{
    if (!next_surface_ && next_step_ != no_intersection())
    {
        // Reset a previously found truncated distance
        this->clear_next_step();
    }

    if (!this->has_next_step())
    {
        detail::LocalState local;
        local.pos        = states_.pos[thread_];
        local.dir        = states_.dir[thread_];
        local.volume     = states_.vol[thread_];
        local.surface    = {states_.surf[thread_], states_.sense[thread_]};
        local.temp_sense = this->make_temp_sense();
        local.temp_next  = this->make_temp_next();

        SimpleUnitTracker tracker(params_);
        auto              isect = tracker.intersect(local);
        next_step_              = isect.distance;
        next_surface_           = isect.surface;
    }

    Propagation result;
    result.distance = next_step_;
    result.boundary = static_cast<bool>(next_surface_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find a nearby distance to the next geometric boundary up to a distance.
 *
 * This may reduce the number of surfaces needed to check, sort, or write to
 * temporary memory, thereby speeding up transport.
 */
CELER_FUNCTION Propagation OrangeTrackView::find_next_step(real_type max_step)
{
    CELER_EXPECT(max_step > 0);

    if (next_step_ > max_step)
    {
        // Cached next step is beyond the given step
        Propagation result;
        result.distance = max_step;
        result.boundary = false;
        return result;
    }
    else if (!next_surface_ && next_step_ < max_step)
    {
        // Reset a previously found truncated distance
        this->clear_next_step();
    }

    if (!this->has_next_step())
    {
        detail::LocalState local;
        local.pos        = states_.pos[thread_];
        local.dir        = states_.dir[thread_];
        local.volume     = states_.vol[thread_];
        local.surface    = {states_.surf[thread_], states_.sense[thread_]};
        local.temp_sense = this->make_temp_sense();
        local.temp_next  = this->make_temp_next();

        SimpleUnitTracker tracker(params_);
        auto              isect = tracker.intersect(local, max_step);
        next_step_              = isect.distance;
        next_surface_           = isect.surface;
    }

    Propagation result;
    result.distance = next_step_;
    result.boundary = static_cast<bool>(next_surface_);

    CELER_ENSURE(result.distance <= max_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the nearest (any direction) boundary within the current volume.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety(const Real3&)
{
    CELER_NOT_IMPLEMENTED("safety distance in ORANGE");
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next straight-line boundary but do not change volume
 */
CELER_FUNCTION void OrangeTrackView::move_to_boundary()
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(next_surface_);

    // Physically move next step
    axpy(next_step_, states_.dir[thread_], &states_.pos[thread_]);
    // Move to the inside of the surface
    states_.surf[thread_]  = next_surface_.id();
    states_.sense[thread_] = next_surface_.unchecked_sense();
    this->clear_next_step();
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume.
 *
 * The straight-line distance *must* be less than the distance to the
 * boundary.
 */
CELER_FUNCTION void OrangeTrackView::move_internal(real_type dist)
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(dist > 0 && dist <= next_step_);
    CELER_EXPECT(dist != next_step_ || !next_surface_);

    // Move and update next_step_
    axpy(dist, states_.dir[thread_], &states_.pos[thread_]);
    next_step_ -= dist;
    states_.surf[thread_] = {};
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * \todo Currently it's up to the caller to make sure that the position is
 * "nearby". We should actually test this with a safety distance.
 */
CELER_FUNCTION void OrangeTrackView::move_internal(const Real3& pos)
{
    states_.pos[thread_]  = pos;
    states_.surf[thread_] = {};
    this->clear_next_step();
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
CELER_FUNCTION void OrangeTrackView::cross_boundary()
{
    CELER_EXPECT(this->surface_id());
    CELER_EXPECT(!this->has_next_step());

    // Flip current sense from "before crossing" to "after"
    detail::LocalState local;
    local.pos     = this->pos();
    local.dir     = this->dir();
    local.volume  = states_.vol[thread_];
    local.surface = {states_.surf[thread_], flip_sense(states_.sense[thread_])};
    local.temp_sense = this->make_temp_sense();

    // Update the post-crossing volume
    SimpleUnitTracker tracker(params_);
    auto              init = tracker.cross_boundary(local);
    // TODO: error correction/graceful failure if initialization failed
    CELER_ASSERT(init.volume);
    states_.vol[thread_]   = init.volume;
    states_.surf[thread_]  = init.surface.id();
    states_.sense[thread_] = init.surface.unchecked_sense();
}

//---------------------------------------------------------------------------//
/*!
 * Change the track's direction.
 *
 * This happens after a scattering event or movement inside a magnetic field.
 * It resets the calculated distance-to-boundary.
 */
CELER_FUNCTION void OrangeTrackView::set_dir(const Real3& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));
    states_.dir[thread_] = newdir;
    this->clear_next_step();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool OrangeTrackView::is_outside() const
{
    // Zeroth volume in outermost universe is always the exterior by
    // construction in ORANGE
    return states_.vol[thread_] == VolumeId{0};
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION Span<Sense> OrangeTrackView::make_temp_sense() const
{
    const auto max_faces = params_.scalars.max_faces;
    auto       offset    = thread_.get() * max_faces;
    return states_.temp_sense[AllItems<Sense, MemSpace::native>{}].subspan(
        offset, max_faces);
}

//---------------------------------------------------------------------------//
/*!
 * Set up intersection scratch space.
 */
CELER_FUNCTION detail::TempNextFace OrangeTrackView::make_temp_next() const
{
    const auto max_isect = params_.scalars.max_intersections;
    auto       offset    = thread_.get() * max_isect;

    detail::TempNextFace result;
    result.face     = states_.temp_face[AllItems<FaceId>{}].data() + offset;
    result.distance = states_.temp_distance[AllItems<real_type>{}].data()
                      + offset;
    result.isect = states_.temp_isect[AllItems<size_type>{}].data() + offset;
    result.size  = max_isect;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Whether any next step has been calculated.
 */
CELER_FUNCTION bool OrangeTrackView::has_next_step() const
{
    return next_step_ != 0;
}

//---------------------------------------------------------------------------//
/*!
 * Reset the next distance-to-boundary.
 *
 * The next surface ID should only ever be used when next_step is zero, so it
 * is OK to wrap it with the CELERITAS_DEBUG conditional.
 */
CELER_FUNCTION void OrangeTrackView::clear_next_step()
{
    next_step_ = 0;
#if CELERITAS_DEBUG
    next_surface_ = {};
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
