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
    // Write out state on destruction
    inline CELER_FUNCTION ~OrangeTrackView();

    // Initialize the state
    inline CELER_FUNCTION OrangeTrackView& operator=(const Initializer_t& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION OrangeTrackView&
                          operator=(const DetailedInitializer& init);

    //// ACCESSORS ////

    //! The current position
    CELER_FUNCTION const Real3& pos() const { return local_.pos; }
    //! The current direction
    CELER_FUNCTION const Real3& dir() const { return local_.dir; }
    //! The current volume ID (null if outside)
    CELER_FUNCTION VolumeId volume_id() const { return local_.volume; }
    //! The current surface ID
    CELER_FUNCTION SurfaceId surface_id() const { return local_.surface.id(); }
    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary
    inline CELER_FUNCTION real_type find_next_step();

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

    detail::LocalState local_;          //!< Temporary local state
    real_type          next_step_{0};   //!< Temporary next step
    detail::OnSurface  next_surface_{}; //!< Temporary next surface
    bool dirty_{false}; //!< Whether global params is updated in destructor

    //// HELPER FUNCTIONS ////

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

    // Set up basic local data
    local_.pos     = states_.pos[thread_];
    local_.dir     = states_.dir[thread_];
    local_.volume  = states_.vol[thread_];
    local_.surface = {states_.surf[thread_], states_.sense[thread_]};

    // Set up sense scratch space
    // TODO: experiment with making this 'lazy'?
    {
        const auto max_faces = params_.scalars.max_faces;
        auto       offset    = thread_.get() * max_faces;
        local_.temp_sense
            = states_.temp_sense[AllItems<Sense, MemSpace::native>{}].subspan(
                offset, max_faces);
    }

    // Set up intersection scratch space
    {
        const auto max_isect = params_.scalars.max_intersections;
        auto       offset    = thread_.get() * max_isect;

        local_.temp_next.face = states_.temp_face[AllItems<FaceId>{}].data()
                                + offset;
        local_.temp_next.distance
            = states_.temp_distance[AllItems<real_type>{}].data() + offset;
        local_.temp_next.isect
            = states_.temp_isect[AllItems<size_type>{}].data() + offset;
        local_.temp_next.size = max_isect;
    }

    CELER_ENSURE(!this->has_next_step());
}

//---------------------------------------------------------------------------//
/*!
 * Write out state on destruction.
 */
CELER_FUNCTION OrangeTrackView::~OrangeTrackView()
{
    if (dirty_)
    {
        states_.pos[thread_]   = local_.pos;
        states_.dir[thread_]   = local_.dir;
        states_.vol[thread_]   = local_.volume;
        states_.surf[thread_]  = local_.surface.id();
        states_.sense[thread_] = local_.surface.unchecked_sense();
    }
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

    local_.pos     = init.pos;
    local_.dir     = init.dir;
    local_.volume  = {};
    local_.surface = {};
    this->clear_next_step();
    dirty_         = true;

    // Initialize logical state
    SimpleUnitTracker tracker(params_);
    auto              tinit = tracker.initialize(local_);
    // TODO: error correction/graceful failure if initialiation failured
    CELER_ASSERT(tinit.volume && !tinit.surface);
    local_.volume = tinit.volume;

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
    CELER_EXPECT(init.other.local_.volume);

    // Copy other track's position but update the direction
    local_.pos     = init.other.local_.pos;
    local_.dir     = init.dir;
    local_.volume  = init.other.local_.volume;
    local_.surface = init.other.local_.surface;

    // Clear step and surface info
    this->clear_next_step();
    dirty_     = true;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION real_type OrangeTrackView::find_next_step()
{
    if (!this->has_next_step())
    {
        SimpleUnitTracker tracker(params_);
        auto              isect = tracker.intersect(local_);
        next_step_              = isect.distance;
        next_surface_           = isect.surface;
    }

    return next_step_;
}

//---------------------------------------------------------------------------//
/*!
 * Find the nearest (any direction) boundary within the current volume.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety(const Real3&)
{
    CELER_NOT_IMPLEMENTED("safety distance in ORANGE");
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
    axpy(next_step_, local_.dir, &local_.pos);
    // Move to the inside of the surface
    local_.surface = next_surface_;
    this->clear_next_step();
    dirty_         = true;
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
    CELER_EXPECT(dist > 0 && dist < next_step_);

    // Move and update next_step_
    axpy(dist, local_.dir, &local_.pos);
    next_step_ -= dist;
    local_.surface = {};
    dirty_ = true;
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
    local_.pos     = pos;
    local_.surface = {};
    this->clear_next_step();
    dirty_         = true;
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
CELER_FUNCTION void OrangeTrackView::cross_boundary()
{
    CELER_EXPECT(local_.surface);
    CELER_EXPECT(!this->has_next_step());

    // Flip current sense from "before crossing" to "after"
    local_.surface.flip_sense();

    // Update the post-crossing volume
    SimpleUnitTracker tracker(params_);
    auto              init = tracker.initialize(local_);
    // TODO: error correction/graceful failure if initialization failed
    CELER_ASSERT(init.volume);
    local_.volume  = init.volume;
    local_.surface = init.surface;
    dirty_         = true;
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
    local_.dir = newdir;
    this->clear_next_step();
    dirty_     = true;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool OrangeTrackView::is_outside() const
{
    // Zeroth volume in outermost universe is always the exterior by
    // construction in ORANGE
    return local_.volume == VolumeId{0};
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
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
