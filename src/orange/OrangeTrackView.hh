//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "Translator.hh"
#include "detail/LevelStateAccessor.hh"
#include "detail/UniverseIndexer.hh"
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
 * - \c find_next_step
 * - \c find_safety or \c move_internal or \c move_to_boundary
 * - if on boundary, \c cross_boundary
 * - at any time, \c set_dir , but then must do \c find_next_step before any
 *   \c move or \c cross action above
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
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    using StateRef = NativeRef<OrangeStateData>;
    using Initializer_t = GeoTrackInitializer;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        OrangeTrackView const& other;  //!< Existing geometry
        Real3 const& dir;  //!< New direction
    };

  public:
    // Construct from params and state
    inline CELER_FUNCTION OrangeTrackView(ParamsRef const& params,
                                          StateRef const& states,
                                          TrackSlotId tid);

    // Initialize the state
    inline CELER_FUNCTION OrangeTrackView& operator=(Initializer_t const& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION OrangeTrackView&
    operator=(DetailedInitializer const& init);

    //// ACCESSORS ////

    // The current position
    CELER_FORCEINLINE_FUNCTION Real3 const& pos() const;
    // The current direction
    CELER_FORCEINLINE_FUNCTION Real3 const& dir() const;
    // The current volume ID (null if outside)
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;
    // The current surface ID
    CELER_FORCEINLINE_FUNCTION SurfaceId surface_id() const;
    // After 'find_next_step', the next straight-line surface
    CELER_FORCEINLINE_FUNCTION SurfaceId next_surface_id() const;
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
    inline CELER_FUNCTION void move_internal(Real3 const& pos);

    // Cross from one side of the current surface to the other
    inline CELER_FUNCTION void cross_boundary();

    // Change direction
    inline CELER_FUNCTION void set_dir(Real3 const& newdir);

  private:
    //// DATA ////

    ParamsRef const& params_;
    StateRef const& states_;
    TrackSlotId track_slot_;

    // Temporary next-step data
    real_type next_step_{0};
    detail::OnSurface next_surface_{};
    LevelId next_surface_level_{};

    //// STATE ASSESSORS ////

    // The current level
    CELER_FORCEINLINE_FUNCTION LevelId& level();

    // The current surface level
    CELER_FORCEINLINE_FUNCTION LevelId& surface_level();

    // The next step distance, as stored on the state
    CELER_FORCEINLINE_FUNCTION real_type& next_step();

    // The next surface to be encounted
    CELER_FORCEINLINE_FUNCTION detail::OnSurface& next_surface();

    // The level of the next surface to be encounted
    CELER_FORCEINLINE_FUNCTION LevelId& next_surface_level();

    //// CONST STATE ASSESSORS ////

    // The current level
    CELER_FORCEINLINE_FUNCTION LevelId const& level() const;

    // The current surface level
    CELER_FORCEINLINE_FUNCTION LevelId const& surface_level() const;

    // The next step distance, as stored on the state
    CELER_FORCEINLINE_FUNCTION real_type const& next_step() const;

    // The next surface to be encounted
    CELER_FORCEINLINE_FUNCTION detail::OnSurface const& next_surface() const;

    // The level of the next surface to be encounted
    CELER_FORCEINLINE_FUNCTION LevelId const& next_surface_level() const;

    //// HELPER FUNCTIONS ////

    // Iterate over layers to find the next step
    inline CELER_FUNCTION void find_next_step_impl(detail::Intersection isect);

    // Create a local tracker
    inline CELER_FUNCTION SimpleUnitTracker make_tracker(UniverseId) const;

    // Create local sense reference
    inline CELER_FUNCTION Span<Sense> make_temp_sense() const;

    // Create local distance
    inline CELER_FUNCTION detail::TempNextFace make_temp_next() const;

    inline CELER_FUNCTION detail::LocalState
    make_local_state(LevelId level) const;

    // Make a LevelStateAccessor for the current thread and level
    CELER_FORCEINLINE_FUNCTION LevelStateAccessor make_lsa() const;

    // Make a LevelStateAccessor for the current thread and a given level
    CELER_FORCEINLINE_FUNCTION LevelStateAccessor make_lsa(LevelId level) const;

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
OrangeTrackView::OrangeTrackView(ParamsRef const& params,
                                 StateRef const& states,
                                 TrackSlotId tid)
    : params_(params), states_(states), track_slot_(tid)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(states_);
    CELER_EXPECT(track_slot_ < states.size());

    this->next_step() = 0;
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
OrangeTrackView::operator=(Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    // Create local state
    detail::LocalState local;
    local.pos = init.pos;
    local.dir = init.dir;
    local.volume = {};
    local.surface = {};
    local.temp_sense = this->make_temp_sense();

    // Recurse into daughter universes starting with the outermost universe
    UniverseId uid = top_universe_id();
    DaughterId daughter_id;
    size_type level = 0;

    do
    {
        auto tracker = this->make_tracker(uid);
        auto tinit = tracker.initialize(local);
        // TODO: error correction/graceful failure if initialiation failed
        CELER_ASSERT(tinit.volume && !tinit.surface);

        auto lsa = this->make_lsa(LevelId{level});
        lsa.vol() = tinit.volume;
        lsa.pos() = local.pos;
        lsa.dir() = local.dir;
        lsa.universe() = uid;
        lsa.surf() = LocalSurfaceId{};
        lsa.sense() = Sense{};
        lsa.boundary() = BoundaryResult::exiting;

        daughter_id = tracker.daughter(tinit.volume);

        if (daughter_id)
        {
            auto const& daughter = params_.daughters[daughter_id];
            auto const& trans = params_.translations[daughter.translation_id];
            TranslatorDown td(trans);
            local.pos = td(local.pos);

            uid = daughter.universe_id;
            ++level;
        }

    } while (daughter_id);

    this->level() = LevelId{level};
    this->surface_level() = LevelId{};

    this->next_step() = 0;
    this->next_surface_level() = LevelId{};

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state from a direction and a copy of the parent state.
 */
CELER_FUNCTION
OrangeTrackView& OrangeTrackView::operator=(DetailedInitializer const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    for (auto i : range(states_.level[init.other.track_slot_] + 1))
    {
        // Copy all data accessed via LSA except for direction
        auto lsa = this->make_lsa(LevelId{i});
        if (this != &init.other)
        {
            lsa = init.other.make_lsa(LevelId{i});
        }
        // TODO: apply rotation when we use Transform instead of Translate
        lsa.dir() = init.dir;
    }

    if (this != &init.other)
    {
        // Copy init track's position but update the direction
        this->level() = states_.level[init.other.track_slot_];
        this->surface_level() = states_.surface_level[init.other.track_slot_];
    }

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * The current position.
 */
CELER_FUNCTION Real3 const& OrangeTrackView::pos() const
{
    return this->make_lsa(LevelId{0}).pos();
}

//---------------------------------------------------------------------------//
/*!
 * The current direction.
 */
CELER_FUNCTION Real3 const& OrangeTrackView::dir() const
{
    return this->make_lsa(LevelId{0}).dir();
}

//---------------------------------------------------------------------------//
/*!
 * The current volume ID (null if outside).
 */
CELER_FUNCTION VolumeId OrangeTrackView::volume_id() const
{
    auto lsa = this->make_lsa();
    detail::UniverseIndexer ui(params_.universe_indexer_data);
    return ui.global_volume(lsa.universe(), lsa.vol());
}

//---------------------------------------------------------------------------//
/*!
 * The current surface ID.
 */
CELER_FUNCTION SurfaceId OrangeTrackView::surface_id() const
{
    if (this->surface_level())
    {
        auto lsa = this->make_lsa(this->surface_level());
        CELER_ASSERT(lsa.surf());
        detail::UniverseIndexer ui(params_.universe_indexer_data);
        return ui.global_surface(lsa.universe(), lsa.surf());
    }
    else
    {
        return SurfaceId{};
    }
}

//---------------------------------------------------------------------------//
/*!
 * After 'find_next_step', the next straight-line surface.
 */
CELER_FUNCTION SurfaceId OrangeTrackView::next_surface_id() const
{
    return this->next_surface().id();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool OrangeTrackView::is_outside() const
{
    // Zeroth volume in outermost universe is always the exterior by
    // construction in ORANGE
    auto lsa = this->make_lsa(LevelId{0});
    return lsa.vol() == LocalVolumeId{0};
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
    auto lsa = this->make_lsa();

    if (CELER_UNLIKELY(lsa.boundary() == BoundaryResult::reentrant))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }

    auto tracker = this->make_tracker(UniverseId{0});
    auto isect = tracker.intersect(this->make_local_state(LevelId{0}));
    this->find_next_step_impl(isect);

    Propagation result;
    result.distance = this->next_step();
    result.boundary = static_cast<bool>(this->next_surface());
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

    auto lsa = this->make_lsa();

    if (CELER_UNLIKELY(lsa.boundary() == BoundaryResult::reentrant))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }
    else if (this->next_step() > max_step)
    {
        // Cached next step is beyond the given step
        return {max_step, false};
    }
    else if (!this->next_surface() && this->next_step() < max_step)
    {
        // Reset a previously found truncated distance
        this->clear_next_step();
    }

    if (!this->has_next_step())
    {
        auto tracker = this->make_tracker(UniverseId{0});
        auto isect
            = tracker.intersect(this->make_local_state(LevelId{0}), max_step);
        this->find_next_step_impl(isect);
    }

    Propagation result;
    result.distance = this->next_step();
    result.boundary = static_cast<bool>(this->next_surface());

    CELER_ENSURE(result.distance <= max_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next straight-line boundary but do not change volume.
 */
CELER_FUNCTION void OrangeTrackView::move_to_boundary()
{
    CELER_EXPECT(this->make_lsa().boundary() != BoundaryResult::reentrant);
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(this->next_surface());

    // Physically move next step
    for (auto i : range(this->level() + 1))
    {
        auto lsa = this->make_lsa(LevelId{i});
        axpy(this->next_step(), lsa.dir(), &lsa.pos());
    }

    // Update the the surface on the applicable level, which the current level
    // or the parent level
    this->surface_level() = this->next_surface_level();

    auto lsa = this->make_lsa(this->surface_level());
    detail::UniverseIndexer ui(params_.universe_indexer_data);
    lsa.surf() = ui.local_surface(this->next_surface_id()).surface;
    lsa.sense() = this->next_surface().unchecked_sense();

    this->clear_next_step();
    CELER_ENSURE(this->is_on_boundary());
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
    CELER_EXPECT(dist > 0 && dist <= this->next_step());
    CELER_EXPECT(dist != this->next_step() || !this->next_surface());

    // Move and update the next step
    for (auto i : range(this->level() + 1))
    {
        auto lsa = this->make_lsa(LevelId{i});
        axpy(dist, lsa.dir(), &lsa.pos());
        lsa.surf() = LocalSurfaceId{};
    }
    this->next_step() -= dist;

    this->surface_level() = LevelId{};
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * \todo Currently it's up to the caller to make sure that the position is
 * "nearby". We should actually test this with a safety distance.
 */
CELER_FUNCTION void OrangeTrackView::move_internal(Real3 const& pos)
{
    auto local_pos = pos;

    for (auto i : range(this->level() + 1))
    {
        auto lsa = this->make_lsa(LevelId{i});
        lsa.pos() = local_pos;
        lsa.surf() = LocalSurfaceId{};

        if (i < this->level())
        {
            auto tracker = this->make_tracker(lsa.universe());
            auto daughter_id = tracker.daughter(lsa.vol());
            CELER_ASSERT(daughter_id);
            auto const& daughter = params_.daughters[daughter_id];
            auto const& trans = params_.translations[daughter.translation_id];

            TranslatorDown td(trans);
            local_pos = td(pos);
        }
    }

    this->surface_level() = LevelId{};
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

    auto sl = this->surface_level();
    auto lsa = this->make_lsa(sl);

    if (CELER_UNLIKELY(lsa.boundary() == BoundaryResult::reentrant))
    {
        if (sl != LevelId{0})
        {
            CELER_NOT_IMPLEMENTED(
                "reentrant surfaces with multilevel ORANGE geometry");
        }

        // Direction changed while on boundary leading to no change in
        // volume/surface. This is logically equivalent to a reflection.
        lsa.boundary() = BoundaryResult::exiting;
        this->next_surface_level() = LevelId{};
        return;
    }

    // Flip current sense from "before crossing" to "after"
    detail::LocalState local;
    local.pos = lsa.pos();
    local.dir = lsa.dir();

    local.volume = lsa.vol();
    local.surface = {lsa.surf(), flip_sense(lsa.sense())};
    local.temp_sense = this->make_temp_sense();

    // Update the post-crossing volume
    auto tracker = this->make_tracker(lsa.universe());
    auto tinit = tracker.cross_boundary(local);

    CELER_ASSERT(tinit.volume);
    if (!CELERITAS_DEBUG && CELER_UNLIKELY(!tinit.volume))
    {
        // Initialization failure on release mode: set to exterior volume
        // rather than segfaulting
        // TODO: error correction or more graceful failure than losing energy
        tinit.volume = LocalVolumeId{0};
        tinit.surface = {};
    }

    lsa.vol() = tinit.volume;
    lsa.surf() = tinit.surface.id();
    lsa.sense() = tinit.surface.unchecked_sense();
    lsa.boundary() = BoundaryResult::exiting;

    // Starting with the current level (i.e., next_surface_level), iterate down
    // into the deepest level
    size_type level = sl.get();
    LocalVolumeId volume_id = tinit.volume;
    auto universe_id = lsa.universe();
    auto daughter_id = tracker.daughter(volume_id);

    while (daughter_id)
    {
        auto daughter = params_.daughters[daughter_id];
        // Get the translator at the parent level, in order to translate into
        // daughter
        TranslatorDown translator(
            params_.translations[daughter.translation_id]);

        // Make the current level the daughter level
        ++level;
        universe_id = daughter.universe_id;
        auto tracker = this->make_tracker(universe_id);

        // Create local state on the daughter level
        local.pos = translator(local.pos);
        local.volume = {};
        local.surface = {};
        local.temp_sense = this->make_temp_sense();

        volume_id = tracker.initialize(local).volume;
        daughter_id = tracker.daughter(volume_id);

        auto lsa = make_lsa(LevelId{level});
        lsa.vol() = volume_id;
        lsa.pos() = local.pos;
        lsa.dir() = local.dir;
        lsa.universe() = universe_id;
        lsa.surf() = LocalSurfaceId{};
        lsa.boundary() = BoundaryResult::exiting;
    }

    this->level() = LevelId{level};

    CELER_ENSURE(this->is_on_boundary());
    this->clear_next_step();
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
CELER_FUNCTION void OrangeTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));

    auto lsa = this->make_lsa();

    if (this->is_on_boundary())
    {
        // Changing direction on a boundary is dangerous, as it could mean we
        // don't leave the volume after all. Evaluate whether the direction
        // dotted with the surface normal changes (i.e. heading from inside to
        // outside or vice versa).
        auto tracker = this->make_tracker(UniverseId{0});
        detail::UniverseIndexer ui(params_.universe_indexer_data);
        const Real3 normal = tracker.normal(
            this->pos(), ui.local_surface(this->surface_id()).surface);

        if ((dot_product(normal, newdir) >= 0)
            != (dot_product(normal, this->dir()) >= 0))
        {
            // The boundary crossing direction has changed! Reverse our plans
            // to change the logical state and move to a new volume.
            lsa.boundary() = flip_boundary(lsa.boundary());
        }
    }

    // Complete direction setting
    lsa.dir() = newdir;

    this->clear_next_step();
}

//---------------------------------------------------------------------------//
// STATE ACCESSORS
//---------------------------------------------------------------------------//
/*!
 * The current level.
 */
CELER_FUNCTION LevelId& OrangeTrackView::level()
{
    return states_.level[track_slot_];
}

/*!
 * The current surface level.
 */
CELER_FUNCTION LevelId& OrangeTrackView::surface_level()
{
    return states_.surface_level[track_slot_];
}

/*!
 * The next step distance.
 */
CELER_FUNCTION real_type& OrangeTrackView::next_step()
{
    return next_step_;
}

/*!
 * The next surface to be encountered.
 */
CELER_FUNCTION detail::OnSurface& OrangeTrackView::next_surface()
{
    return next_surface_;
}

/*!
 * The level of the next surface to be encounted.
 */
CELER_FUNCTION LevelId& OrangeTrackView::next_surface_level()
{
    return next_surface_level_;
}

//---------------------------------------------------------------------------//
// CONST STATE ACCESSORS
//---------------------------------------------------------------------------//
/*!
 * The current level.
 */
CELER_FUNCTION LevelId const& OrangeTrackView::level() const
{
    return states_.level[track_slot_];
}

/*!
 * The current surface level.
 */
CELER_FUNCTION LevelId const& OrangeTrackView::surface_level() const
{
    return states_.surface_level[track_slot_];
}

/*!
 * The next step distance.
 */
CELER_FUNCTION real_type const& OrangeTrackView::next_step() const
{
    return next_step_;
}

/*!
 * The next surface to be encountered.
 */
CELER_FUNCTION detail::OnSurface const& OrangeTrackView::next_surface() const
{
    return next_surface_;
}

/*!
 * The level of the next surface to be encounted.
 */
CELER_FUNCTION LevelId const& OrangeTrackView::next_surface_level() const
{
    return next_surface_level_;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Iterate over levels 1 to N to find the next step.
 *
 * Caller is responsible for finding the canidate next step on level 0, and
 * passing the resultant Intersection object as an argument
 */
CELER_FUNCTION void
OrangeTrackView::find_next_step_impl(detail::Intersection isect)
{
    // The LevelId corresponding to the level with with minium
    // distance to intersection
    LevelId min_level{0};

    // Find the nearest intersection from level 0 to current level inclusive,
    // prefering the higher level (i.e., lowest uid)
    for (auto levelid : range(LevelId{1}, this->level() + 1))
    {
        auto lsa = this->make_lsa(levelid);
        auto tracker = this->make_tracker(lsa.universe());
        auto local_isect = tracker.intersect(this->make_local_state(levelid),
                                             isect.distance);
        if (local_isect.distance < isect.distance)
        {
            isect = local_isect;
            min_level = levelid;
        }
    }

    this->next_step() = isect.distance;

    // If there is a valid next surface, convert it from local to global
    if (isect)
    {
        detail::UniverseIndexer ui(params_.universe_indexer_data);
        this->next_surface() = celeritas::detail::OnSurface(
            ui.global_surface(this->make_lsa(min_level).universe(),
                              isect.surface.id()),
            isect.surface.unchecked_sense());
        this->next_surface_level() = min_level;
    }
    else
    {
        this->next_surface() = {};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the nearest boundary in any direction.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety()
{
    auto lsa = this->make_lsa();

    if (lsa.surf())
    {
        // Zero distance to boundary on a surface
        return real_type{0};
    }

    CELER_ASSERT(lsa.universe() == UniverseId{0});
    auto tracker = this->make_tracker(lsa.universe());
    return tracker.safety(lsa.pos(), lsa.vol());
}

//---------------------------------------------------------------------------//
/*!
 * Create a local tracker for a universe.
 *
 * \todo Template on tracker type, allow multiple universe types (see
 * UniverseTypeTraits.hh)
 */
CELER_FUNCTION SimpleUnitTracker OrangeTrackView::make_tracker(UniverseId id) const
{
    CELER_EXPECT(id < params_.universe_types.size());
    CELER_EXPECT(id.unchecked_get() == params_.universe_indices[id]);

    using TraitsT = UniverseTypeTraits<UniverseType::simple>;
    using IdT = OpaqueId<typename TraitsT::record_type>;
    using TrackerT = typename TraitsT::tracker_type;

    return TrackerT{params_, IdT{id.unchecked_get()}};
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION Span<Sense> OrangeTrackView::make_temp_sense() const
{
    auto const max_faces = params_.scalars.max_faces;
    auto offset = track_slot_.get() * max_faces;
    return states_.temp_sense[AllItems<Sense, MemSpace::native>{}].subspan(
        offset, max_faces);
}

//---------------------------------------------------------------------------//
/*!
 * Set up intersection scratch space.
 */
CELER_FUNCTION detail::TempNextFace OrangeTrackView::make_temp_next() const
{
    auto const max_isect = params_.scalars.max_intersections;
    auto offset = track_slot_.get() * max_isect;

    detail::TempNextFace result;
    result.face = states_.temp_face[AllItems<FaceId>{}].data() + offset;
    result.distance = states_.temp_distance[AllItems<real_type>{}].data()
                      + offset;
    result.isect = states_.temp_isect[AllItems<size_type>{}].data() + offset;
    result.size = max_isect;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a local state.
 */
CELER_FUNCTION detail::LocalState
OrangeTrackView::make_local_state(LevelId level) const
{
    detail::LocalState local;

    auto lsa = this->make_lsa(level);

    local.pos = lsa.pos();
    local.dir = lsa.dir();
    local.volume = lsa.vol();
    local.surface = {lsa.surf(), lsa.sense()};
    local.temp_sense = this->make_temp_sense();
    local.temp_next = this->make_temp_next();
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
}

//---------------------------------------------------------------------------//
/*!
 * Make a LevelStateAccessor for the current thread and level.
 */
CELER_FUNCTION LevelStateAccessor OrangeTrackView::make_lsa() const
{
    return this->make_lsa(this->level());
}

//---------------------------------------------------------------------------//
/*!
 * Make a LevelStateAccessor for the current thread and a given level.
 */
CELER_FUNCTION LevelStateAccessor OrangeTrackView::make_lsa(LevelId level) const
{
    return LevelStateAccessor(&states_, track_slot_, level);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
