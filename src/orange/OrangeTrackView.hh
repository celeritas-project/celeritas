//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/sys/ThreadId.hh"

#include "OrangeData.hh"
#include "OrangeTypes.hh"
#include "detail/UnitIndexer.hh"
#include "univ/SimpleUnitTracker.hh"
#include "univ/UniverseTypeTraits.hh"
#include "univ/detail/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Navigate through an ORANGE geometry on a single thread.
 *
 * Ordering requirements:
 * - initialize (through assignment) must come first
 * - access (pos, dir, volume/surface/is_outside/is_on_boundary) good at any
 * time
 * - \c find_safety (fine at any time)
 * - \c find_next_step
 * - \c move_internal or \c move_to_boundary
 * - if on boundary, \c cross_boundary
 * - at any time, \c set_dir , but then must do \c find_next_step before any
 *   following action above
 *
 * The main point is that \c find_next_step depends on the current
 * straight-line direction, \c move_to_boundary and \c move_internal (with
 * a step length) depends on that distance, and
 * \c cross_boundary depends on being on the boundary with a knowledge of the
 * post-boundary state.
 *
 * \c move_internal with a position \em should depend on the safety distance
 * but that's not yet implemented.
 */
class OrangeTrackView
{
  public:
    //!@{
    //! Type aliases
    using ParamsRef     = NativeCRef<OrangeParamsData>;
    using StateRef      = NativeRef<OrangeStateData>;
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
    //! After 'find_next_step', the next straight-line surface
    CELER_FUNCTION SurfaceId next_surface_id() const
    {
        return next_surface_.id();
    }
    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    CELER_FORCEINLINE_FUNCTION bool is_on_boundary() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary
    inline CELER_FUNCTION Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the distance to the nearest boundary in any direction
    inline CELER_FUNCTION real_type find_safety();

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

    real_type         next_step_{0};   //!< Temporary next step
    detail::OnSurface next_surface_{}; //!< Temporary next surface

    //// HELPER FUNCTIONS ////

    // Create a local tracker
    inline CELER_FUNCTION SimpleUnitTracker make_tracker(UniverseId) const;

    // Create local sense reference
    inline CELER_FUNCTION Span<Sense> make_temp_sense() const;

    // Create local distance
    inline CELER_FUNCTION detail::TempNextFace make_temp_next() const;

    inline CELER_FUNCTION detail::LocalState make_local_state() const;

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
    states_.pos[thread_]      = init.pos;
    states_.dir[thread_]      = init.dir;
    states_.surf[thread_]     = {};
    states_.sense[thread_]    = {};
    states_.boundary[thread_] = BoundaryResult::exiting;

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
    UniverseId          uid = base_universe_id();
    VolumeId            global_vol_id{};
    detail::UnitIndexer unit_indexer(params_.unit_indexer_data);

    // Recurse into daughter universes starting with the outermost universe
    do
    {
        auto tracker = this->make_tracker(uid);
        auto tinit   = tracker.initialize(local);
        // TODO: error correction/graceful failure if initialiation failed
        CELER_ASSERT(tinit.volume && !tinit.surface);
        global_vol_id = unit_indexer.global_volume(uid, tinit.volume);

        OpaqueId<VolumeRecord> vrid(global_vol_id.unchecked_get());
        uid = params_.volume_records[vrid].daughter;
    } while (uid);

    states_.vol[thread_] = global_vol_id;

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
    states_.pos[thread_]      = states_.pos[init.other.thread_];
    states_.dir[thread_]      = init.dir;
    states_.vol[thread_]      = states_.vol[init.other.thread_];
    states_.surf[thread_]     = states_.surf[init.other.thread_];
    states_.sense[thread_]    = states_.sense[init.other.thread_];
    states_.boundary[thread_] = states_.boundary[init.other.thread_];

    // Clear step and surface info
    this->clear_next_step();

    CELER_ENSURE(!this->has_next_step());
    return *this;
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
/*!
 * Whether the track is exactly on a surface.
 */
CELER_FUNCTION bool OrangeTrackView::is_on_boundary() const
{
    return static_cast<bool>(this->surface_id());
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation OrangeTrackView::find_next_step()
{
    if (CELER_UNLIKELY(states_.boundary[thread_] == BoundaryResult::reentrant))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }

    if (!next_surface_ && next_step_ != no_intersection())
    {
        // Reset a previously found truncated distance
        this->clear_next_step();
    }

    if (!this->has_next_step())
    {
        auto tracker  = this->make_tracker(UniverseId{0});
        auto isect    = tracker.intersect(this->make_local_state());
        next_step_    = isect.distance;
        next_surface_ = isect.surface;
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

    if (CELER_UNLIKELY(states_.boundary[thread_] == BoundaryResult::reentrant))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }
    else if (next_step_ > max_step)
    {
        // Cached next step is beyond the given step
        return {max_step, false};
    }
    else if (!next_surface_ && next_step_ < max_step)
    {
        // Reset a previously found truncated distance
        this->clear_next_step();
    }

    if (!this->has_next_step())
    {
        auto tracker  = this->make_tracker(UniverseId{0});
        auto isect    = tracker.intersect(this->make_local_state(), max_step);
        next_step_    = isect.distance;
        next_surface_ = isect.surface;
    }

    Propagation result;
    result.distance = next_step_;
    result.boundary = static_cast<bool>(next_surface_);

    CELER_ENSURE(result.distance <= max_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the nearest boundary in any direction.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety()
{
    if (states_.surf[thread_])
    {
        // Zero distance to boundary on a surface
        return real_type{0};
    }

    auto tracker = this->make_tracker(UniverseId{0});
    return tracker.safety(this->pos(), this->volume_id());
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next straight-line boundary but do not change volume
 */
CELER_FUNCTION void OrangeTrackView::move_to_boundary()
{
    CELER_EXPECT(states_.boundary[thread_] != BoundaryResult::reentrant);
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
 * The position *must* be on the boundary following a move-to-boundary. This
 * should only be called once per boundary crossing.
 */
CELER_FUNCTION void OrangeTrackView::cross_boundary()
{
    CELER_EXPECT(this->is_on_boundary());
    CELER_EXPECT(!this->has_next_step());

    if (CELER_UNLIKELY(states_.boundary[thread_] == BoundaryResult::reentrant))
    {
        // Direction changed while on boundary leading to no change in
        // volume/surface. This is logically equivalent to a reflection.
        states_.boundary[thread_] = BoundaryResult::exiting;
        return;
    }

    // Flip current sense from "before crossing" to "after"
    detail::LocalState local;
    local.pos     = this->pos();
    local.dir     = this->dir();
    local.volume  = states_.vol[thread_];
    local.surface = {states_.surf[thread_], flip_sense(states_.sense[thread_])};
    local.temp_sense = this->make_temp_sense();

    // Update the post-crossing volume
    auto tracker = this->make_tracker(UniverseId{0});
    auto init    = tracker.cross_boundary(local);
    CELER_ASSERT(init.volume);
    if (!CELERITAS_DEBUG && CELER_UNLIKELY(!init.volume))
    {
        // Initialization failure on release mode: set to exterior volume
        // rather than segfaulting
        // TODO: error correction or more graceful failure than losing energy
        init.volume  = VolumeId{0};
        init.surface = {};
    }
    states_.vol[thread_]   = init.volume;
    states_.surf[thread_]  = init.surface.id();
    states_.sense[thread_] = init.surface.unchecked_sense();

    // Reset boundary crossing state
    states_.boundary[thread_] = BoundaryResult::exiting;

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Change the track's direction.
 *
 * This happens after a scattering event or movement inside a magnetic field.
 * It resets the calculated distance-to-boundary. It is allowed to happen on
 * the boundary, but changing direction so that it goes from pointing outward
 * to inward (or vice versa) will mean that \c cross_boundary will be a
 * null-op.
 */
CELER_FUNCTION void OrangeTrackView::set_dir(const Real3& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));

    if (this->is_on_boundary())
    {
        // Changing direction on a boundary is dangerous, as it could mean we
        // don't leave the volume after all. Evaluate whether the direction
        // dotted with the surface normal changes (i.e. heading from inside to
        // outside or vice versa).
        auto        tracker = this->make_tracker(UniverseId{0});
        const Real3 normal  = tracker.normal(this->pos(), this->surface_id());

        if ((dot_product(normal, newdir) >= 0)
            != (dot_product(normal, this->dir()) >= 0))
        {
            // The boundary crossing direction has changed! Reverse our plans
            // to change the logical state and move to a new volume.
            states_.boundary[thread_]
                = flip_boundary(states_.boundary[thread_]);
        }
    }

    // Complete direction setting
    states_.dir[thread_] = newdir;

    this->clear_next_step();
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Create a local tracker for a universe.
 *
 * \todo Template on tracker type, allow multiple universe types (see
 * UniverseTypeTraits.hh)
 */
CELER_FUNCTION SimpleUnitTracker OrangeTrackView::make_tracker(UniverseId id) const
{
    CELER_EXPECT(id < params_.universe_type.size());
    CELER_EXPECT(id.unchecked_get() == params_.universe_index[id]);

    using TraitsT  = UniverseTypeTraits<UniverseType::simple>;
    using IdT      = OpaqueId<typename TraitsT::record_type>;
    using TrackerT = typename TraitsT::tracker_type;

    return TrackerT{params_, IdT{id.unchecked_get()}};
}

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
 * Create a local state.
 */
CELER_FUNCTION detail::LocalState OrangeTrackView::make_local_state() const
{
    detail::LocalState local;
    local.pos        = states_.pos[thread_];
    local.dir        = states_.dir[thread_];
    local.volume     = states_.vol[thread_];
    local.surface    = {states_.surf[thread_], states_.sense[thread_]};
    local.temp_sense = this->make_temp_sense();
    local.temp_next  = this->make_temp_next();
    return local;
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
