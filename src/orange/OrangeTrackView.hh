//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/sys/ThreadId.hh"

#include "LevelStateAccessor.hh"
#include "OrangeData.hh"
#include "OrangeTypes.hh"
#include "transform/TransformVisitor.hh"
#include "univ/SimpleUnitTracker.hh"
#include "univ/TrackerVisitor.hh"
#include "univ/UniverseTypeTraits.hh"
#include "univ/detail/Types.hh"

#include "detail/UniverseIndexer.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "corecel/io/Repr.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Navigate through an ORANGE geometry on a single thread.
 *
 * Since the navigation relies on computationally expensive calls and must
 * ensure a consistent state between physical and logical boundaries, there is
 * an ordering followed by Celeritas' internal calls to each track's state.
 * Access (\c pos, \c dir, \c volume/surface/is_outside/is_on_boundary) is
 * valid at any time.
 *
 * The required ordering is:
 *
 * - Initialization, via the assignment operator
 * - Locating the boundary crossing along the current direction, \c
 *   find_next_step
 * - Locating the closest point on the boundary in any direction, \c
 *   find_safety
 * - Movement within a volume, not crossing a boundary: \c move_internal or \c
 *   move_to_boundary
 * - If on a boundary, logically moving to the adjacent volume ("relocation")
 *   with \c cross_boundary
 *
 * At any time, \c set_dir may be called, but then \c find_next_step must again
 * be called before any subsequent \c move or \c cross action above .
 *
 * The main point is that \c find_next_step depends on the current
 * straight-line direction, \c move_to_boundary and \c move_internal (with
 * a step length) depends on that distance, and
 * \c cross_boundary depends on being on the boundary with a knowledge of the
 * post-boundary state.
 *
 * \todo \c move_internal with a position \em should depend on the safety
 * distance, but that check is not yet implemented.
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
    inline CELER_FUNCTION Real3 const& pos() const;
    // The current direction
    inline CELER_FUNCTION Real3 const& dir() const;

    // The current volume ID (null if outside)
    inline CELER_FUNCTION VolumeId volume_id() const;
    // Get the physical volume ID in the current cell
    inline CELER_FUNCTION VolumeInstanceId volume_instance_id() const;
    // The current level
    inline CELER_FUNCTION LevelId const& level() const;
    // Get the volume instance ID for all levels
    inline CELER_FUNCTION void volume_instance_id(Span<VolumeInstanceId>) const;

    // The current surface ID
    inline CELER_FUNCTION SurfaceId surface_id() const;
    // After 'find_next_step', the next straight-line surface
    inline CELER_FUNCTION SurfaceId next_surface_id() const;
    // Whether the track is outside the valid geometry region
    inline CELER_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    inline CELER_FUNCTION bool is_on_boundary() const;
    //! Whether the last operation resulted in an error
    CELER_FORCEINLINE_FUNCTION bool failed() const { return failed_; }

    //// OPERATIONS ////

    // Find the distance to the next boundary
    inline CELER_FUNCTION Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the distance to the nearest boundary in any direction
    inline CELER_FUNCTION real_type find_safety();

    // Find the distance to the nearest nearby boundary in any direction
    inline CELER_FUNCTION real_type find_safety(real_type max_step);

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
    //// TYPES ////

    using LSA = LevelStateAccessor;

    //// DATA ////

    ParamsRef const& params_;
    StateRef const& states_;
    TrackSlotId track_slot_;
    bool failed_{false};

    //// PRIVATE STATE MUTATORS ////

    // The current level
    inline CELER_FUNCTION void level(LevelId);

    // The boundary on the current surface level
    inline CELER_FUNCTION void boundary(BoundaryResult);

    // The next step distance, as stored on the state
    inline CELER_FUNCTION void next_step(real_type dist);

    // The next surface to be encounted
    inline CELER_FUNCTION void next_surf(detail::OnLocalSurface const&);

    // The level of the next surface to be encounted
    inline CELER_FUNCTION void next_surface_level(LevelId);

    //// PRIVATE STATE ACCESSORS ////

    // The current surface level
    inline CELER_FUNCTION LevelId const& surface_level() const;

    // The local surface on the current surface level
    inline CELER_FUNCTION LocalSurfaceId const& surf() const;

    // The sense on the current surface level
    inline CELER_FUNCTION Sense const& sense() const;

    // The boundary on the current surface level
    inline CELER_FUNCTION BoundaryResult const& boundary() const;

    // The next step distance, as stored on the state
    inline CELER_FUNCTION real_type const& next_step() const;

    // The next surface to be encounted
    inline CELER_FUNCTION detail::OnLocalSurface next_surf() const;

    // The level of the next surface to be encounted
    inline CELER_FUNCTION LevelId const& next_surface_level() const;

    //// HELPER FUNCTIONS ////

    // Iterate over lower levels to find the next step
    inline CELER_FUNCTION Propagation
    find_next_step_impl(detail::Intersection isect);

    // Create local sense reference
    inline CELER_FUNCTION Span<SenseValue> make_temp_sense() const;

    // Create local distance
    inline CELER_FUNCTION detail::TempNextFace make_temp_next() const;

    inline CELER_FUNCTION detail::LocalState
    make_local_state(LevelId level) const;

    // Whether the next distance-to-boundary has been found
    inline CELER_FUNCTION bool has_next_step() const;

    // Whether the next surface has been found
    inline CELER_FUNCTION bool has_next_surface() const;

    // Invalidate the next distance-to-boundary and surface
    inline CELER_FUNCTION void clear_next();

    // Assign the surface on the current level
    inline CELER_FUNCTION void
    surface(LevelId level, detail::OnLocalSurface surf);

    // Clear the surface on the current level
    inline CELER_FUNCTION void clear_surface();

    // Make a LevelStateAccessor for the current thread and level
    inline CELER_FUNCTION LSA make_lsa() const;

    // Make a LevelStateAccessor for the current thread and a given level
    inline CELER_FUNCTION LSA make_lsa(LevelId level) const;

    // Get the daughter ID for the volume in the universe (or null)
    inline CELER_FUNCTION DaughterId get_daughter(LSA const& lsa);

    // Get the transform ID for the given daughter.
    inline CELER_FUNCTION TransformId get_transform(DaughterId daughter_id);

    // Get the transform ID to move from this level to the one below.
    inline CELER_FUNCTION TransformId get_transform(LevelId lev);
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

    failed_ = false;

    // Create local state
    detail::LocalState local;
    local.pos = init.pos;
    local.dir = init.dir;
    local.volume = {};
    local.surface = {};
    local.temp_sense = this->make_temp_sense();

    // Helpers for applying parent-to-daughter transformations
    TransformVisitor apply_transform{params_};
    auto transform_down_local = [&local](auto&& t) {
        local.pos = t.transform_down(local.pos);
        local.dir = t.rotate_down(local.dir);
    };

    // Recurse into daughter universes starting with the outermost universe
    UniverseId univ_id = top_universe_id();
    DaughterId daughter_id;
    LevelId level{0};
    do
    {
        TrackerVisitor visit_tracker{params_};
        auto tinit = visit_tracker(
            [&local](auto&& t) { return t.initialize(local); }, univ_id);

        if (!tinit.volume || tinit.surface)
        {
#if !CELER_DEVICE_COMPILE
            auto msg = CELER_LOG_LOCAL(error);
            msg << "Failed to initialize geometry state: ";
            if (!tinit.volume)
            {
                msg << "could not find associated volume";
            }
            else
            {
                msg << "started on a surface ("
                    << tinit.surface.id().unchecked_get() << ")";
            }
            msg << " in universe " << univ_id.unchecked_get()
                << " at local position " << repr(local.pos);
#endif
            // Mark as failed and place in local "exterior" to end the search
            // but preserve the current level information
            failed_ = true;
            tinit.volume = orange_exterior_volume;
        }

        auto lsa = this->make_lsa(level);
        lsa.vol() = tinit.volume;
        lsa.pos() = local.pos;
        lsa.dir() = local.dir;
        lsa.universe() = univ_id;

        daughter_id = visit_tracker(
            [&tinit](auto&& t) { return t.daughter(tinit.volume); }, univ_id);

        if (daughter_id)
        {
            auto const& daughter = params_.daughters[daughter_id];
            // Apply "transform down" based on stored transform
            apply_transform(transform_down_local, daughter.trans_id);
            // Update universe and increase level depth
            univ_id = daughter.universe_id;
            ++level;
        }

    } while (daughter_id);

    // Save found level
    this->level(level);

    // Reset surface/boundary information
    this->boundary(BoundaryResult::exiting);
    this->clear_surface();
    this->clear_next();

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

    failed_ = false;

    if (this != &init.other)
    {
        // Copy init track's position and logical state
        this->level(states_.level[init.other.track_slot_]);
        this->surface(init.other.surface_level(),
                      {init.other.surf(), init.other.sense()});
        this->boundary(init.other.boundary());

        for (auto lev : range(LevelId{this->level() + 1}))
        {
            // Copy all data accessed via LSA
            auto lsa = this->make_lsa(lev);
            lsa = init.other.make_lsa(lev);
        }
    }

    // Clear the next step information since we're changing direction or
    // initializing a new state
    this->clear_next();

    // Transform direction from global to local
    Real3 localdir = init.dir;
    auto apply_transform = TransformVisitor{params_};
    auto rotate_down
        = [&localdir](auto&& t) { localdir = t.rotate_down(localdir); };
    for (auto lev : range(LevelId{this->level()}))
    {
        auto lsa = this->make_lsa(lev);
        lsa.dir() = localdir;
        apply_transform(rotate_down,
                        this->get_transform(this->get_daughter(lsa)));
    }

    // Save final level
    this->make_lsa().dir() = localdir;

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
 * The current volume ID.
 *
 * \note It is allowable to call this function when "outside", because the
 * outside in ORANGE is just a special volume. Other geometries may not have
 * that behavior.
 */
CELER_FUNCTION VolumeId OrangeTrackView::volume_id() const
{
    auto lsa = this->make_lsa();
    detail::UniverseIndexer ui(params_.universe_indexer_data);
    return ui.global_volume(lsa.universe(), lsa.vol());
}

//---------------------------------------------------------------------------//
/*!
 * The current volume instance.
 *
 * \todo not implemented; VolumeId is already halfway between a "reusable
 * volume" and a "volume instance" anyway...
 */
CELER_FUNCTION VolumeInstanceId OrangeTrackView::volume_instance_id() const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * The current level.
 */
CELER_FORCEINLINE_FUNCTION LevelId const& OrangeTrackView::level() const
{
    return states_.level[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance ID at every level.
 *
 * The input span size must be equal to the value of "level" plus one. The
 * top-most level ("world" or level zero) starts at index zero and moves
 * downward. Note that Geant4 uses the \em reverse nomenclature.
 */
CELER_FUNCTION void
OrangeTrackView::volume_instance_id(Span<VolumeInstanceId> levels) const
{
    CELER_EXPECT(levels.size() == this->level().get() + 1);
    for (auto lev : range(levels.size()))
    {
        levels[lev] = {};
    }
}

//---------------------------------------------------------------------------//
/*!
 * The current surface ID.
 */
CELER_FUNCTION SurfaceId OrangeTrackView::surface_id() const
{
    if (this->is_on_boundary())
    {
        auto lsa = this->make_lsa(this->surface_level());
        detail::UniverseIndexer ui{params_.universe_indexer_data};
        return ui.global_surface(lsa.universe(), this->surf());
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
    CELER_EXPECT(this->has_next_surface());
    auto lsa = this->make_lsa(this->next_surface_level());
    detail::UniverseIndexer ui{params_.universe_indexer_data};
    return ui.global_surface(lsa.universe(), this->next_surf().id());
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
    return lsa.vol() == orange_exterior_volume;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is exactly on a surface.
 */
CELER_FORCEINLINE_FUNCTION bool OrangeTrackView::is_on_boundary() const
{
    return static_cast<bool>(this->surface_level());
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation OrangeTrackView::find_next_step()
{
    if (CELER_UNLIKELY(this->boundary() == BoundaryResult::reentrant))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }

    // Find intersection at the top level: always the first simple unit
    auto global_isect = [this] {
        SimpleUnitTracker t{params_, SimpleUnitId{0}};
        return t.intersect(this->make_local_state(LevelId{0}));
    }();
    // Find intersection for all lower levels
    return this->find_next_step_impl(global_isect);
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

    if (CELER_UNLIKELY(this->boundary() == BoundaryResult::reentrant))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }

    // Find intersection at the top level: always the first simple unit
    auto global_isect = [this, &max_step] {
        SimpleUnitTracker t{params_, SimpleUnitId{0}};
        return t.intersect(this->make_local_state(LevelId{0}), max_step);
    }();
    // Find intersection for all lower levels
    auto result = this->find_next_step_impl(global_isect);
    CELER_ENSURE(result.distance <= max_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next straight-line boundary but do not change volume.
 */
CELER_FUNCTION void OrangeTrackView::move_to_boundary()
{
    CELER_EXPECT(this->boundary() != BoundaryResult::reentrant);
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(this->has_next_surface());

    // Physically move next step
    real_type const dist = this->next_step();
    for (auto i : range(this->level() + 1))
    {
        auto lsa = this->make_lsa(LevelId{i});
        axpy(dist, lsa.dir(), &lsa.pos());
    }

    this->surface(this->next_surface_level(), this->next_surf());
    this->clear_next();
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
    CELER_EXPECT(dist != this->next_step() || !this->has_next_surface());

    // Move and update the next step
    for (auto i : range(this->level() + 1))
    {
        auto lsa = this->make_lsa(LevelId{i});
        axpy(dist, lsa.dir(), &lsa.pos());
    }
    this->next_step(this->next_step() - dist);
    this->clear_surface();
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * \todo Currently it's up to the caller to make sure that the position is
 * "nearby". We should actually test this with an "is inside" call.
 */
CELER_FUNCTION void OrangeTrackView::move_internal(Real3 const& pos)
{
    // Transform all nonlocal levels
    auto local_pos = pos;
    auto apply_transform = TransformVisitor{params_};
    auto translate_down
        = [&local_pos](auto&& t) { local_pos = t.transform_down(local_pos); };
    for (auto lev : range(LevelId{this->level()}))
    {
        auto lsa = this->make_lsa(lev);
        lsa.pos() = local_pos;

        // Apply "transform down" based on stored transform
        apply_transform(translate_down,
                        this->get_transform(this->get_daughter(lsa)));
    }

    // Save final level
    auto lsa = this->make_lsa();
    lsa.pos() = local_pos;

    // Clear surface state and next-step info
    this->clear_surface();
    this->clear_next();
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

    if (CELER_UNLIKELY(this->boundary() == BoundaryResult::reentrant))
    {
        // Direction changed while on boundary leading to no change in
        // volume/surface. This is logically equivalent to a reflection.
        this->boundary(BoundaryResult::exiting);
        return;
    }

    // Cross surface by flipping the sense
    states_.sense[track_slot_] = flip_sense(this->sense());
    this->boundary(BoundaryResult::exiting);

    // Create local state from post-crossing level and updated sense
    LevelId level{this->surface_level()};
    UniverseId universe;
    LocalVolumeId volume;
    detail::LocalState local;
    {
        auto lsa = this->make_lsa(level);
        universe = lsa.universe();
        local.pos = lsa.pos();
        local.dir = lsa.dir();
        local.volume = lsa.vol();
        local.surface = {this->surf(), this->sense()};
        local.temp_sense = this->make_temp_sense();
    }

    TrackerVisitor visit_tracker{params_};

    // Update the post-crossing volume by crossing the boundary of the "surface
    // crossing" level universe
    volume = visit_tracker(
        [&local](auto&& t) { return t.cross_boundary(local).volume; },
        universe);
    if (CELER_UNLIKELY(!volume))
    {
        // Boundary crossing failure
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(error)
            << "track failed to cross local surface "
            << this->surf().unchecked_get() << " in universe "
            << universe.unchecked_get() << " at local position "
            << repr(local.pos) << " along local direction " << repr(local.dir);
#endif
        // Mark as failed and place in local "exterior" to end the search
        // but preserve the current level information
        failed_ = true;
        volume = orange_exterior_volume;
    }
    make_lsa(level).vol() = volume;

    // Clear local surface before diving into daughters
    // TODO: this is where we'd do inter-universe mapping
    local.volume = {};
    local.surface = {};

    // Starting with the current level (i.e., next_surface_level), iterate
    // down into the deepest level: *initializing* not *crossing*
    auto daughter_id = visit_tracker(
        [volume](auto&& t) { return t.daughter(volume); }, universe);
    while (daughter_id)
    {
        ++level;
        {
            // Update universe, local position/direction
            auto const& daughter = params_.daughters[daughter_id];
            TransformVisitor apply_transform{params_};
            auto transform_down_local = [&local](auto&& t) {
                local.pos = t.transform_down(local.pos);
                local.dir = t.rotate_down(local.dir);
            };
            apply_transform(transform_down_local, daughter.trans_id);
            universe = daughter.universe_id;
        }

        // Initialize in daughter and get IDs of volume and potential daughter
        volume = visit_tracker(
            [&local](auto&& t) { return t.initialize(local).volume; },
            universe);

        if (!volume)
        {
#if !CELER_DEVICE_COMPILE
            auto msg = CELER_LOG_LOCAL(error);
            msg << "track failed to cross boundary: could not find associated "
                   "volume in universe "
                << universe.unchecked_get() << " at local position "
                << repr(local.pos);
#endif
            // Mark as failed and place in local "exterior" to end the search
            // but preserve the current level information
            failed_ = true;
            volume = orange_exterior_volume;
        }
        daughter_id = visit_tracker(
            [volume](auto&& t) { return t.daughter(volume); }, universe);

        auto lsa = make_lsa(level);
        lsa.vol() = volume;
        lsa.pos() = local.pos;
        lsa.dir() = local.dir;
        lsa.universe() = universe;
    }

    // Save final level
    this->level(level);

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
 *
 * TODO: This needs to be updated to handle reflections through levels
 */
CELER_FUNCTION void OrangeTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));

    if (this->is_on_boundary())
    {
        // Changing direction on a boundary, which may result in not leaving
        // current volume upon the cross_surface call
        auto lsa = this->make_lsa(this->surface_level());

        TrackerVisitor visit_tracker{params_};
        auto normal = visit_tracker(
            [pos = lsa.pos(), local_surface = this->surf()](auto&& t) {
                return t.normal(pos, local_surface);
            },
            lsa.universe());

        // Normal is in *local* coordinates but newdir is in *global*: rotate
        // up to check
        auto apply_transform = TransformVisitor{params_};
        auto rotate_up = [&normal](auto&& t) { normal = t.rotate_up(normal); };
        for (auto level : range<int>(this->level().unchecked_get()).step(-1))
        {
            apply_transform(rotate_up, this->get_transform(LevelId(level)));
        }

        // Evaluate whether the direction dotted with the surface normal
        // changes (i.e. heading from inside to outside or vice versa).
        if ((dot_product(normal, newdir) >= 0)
            != (dot_product(normal, this->dir()) >= 0))
        {
            // The boundary crossing direction has changed! Reverse our
            // plans to change the logical state and move to a new volume.
            this->boundary(flip_boundary(this->boundary()));
        }
    }

    // Complete direction setting by transforming direction all the way down
    Real3 localdir = newdir;
    auto apply_transform = TransformVisitor{params_};
    auto rotate_down
        = [&localdir](auto&& t) { localdir = t.rotate_down(localdir); };
    for (auto lev : range(this->level()))
    {
        auto lsa = this->make_lsa(lev);
        lsa.dir() = localdir;
        apply_transform(rotate_down,
                        this->get_transform(this->get_daughter(lsa)));
    }
    // Save final level
    this->make_lsa().dir() = localdir;

    this->clear_next();
}

//---------------------------------------------------------------------------//
// STATE ACCESSORS
//---------------------------------------------------------------------------//
/*!
 * The current level.
 */
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::level(LevelId lev)
{
    states_.level[track_slot_] = lev;
}

/*!
 * The boundary on the current surface level.
 */
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::boundary(BoundaryResult br)
{
    states_.boundary[track_slot_] = br;
}

/*!
 * The next step distance.
 */
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::next_step(real_type dist)
{
    states_.next_step[track_slot_] = dist;
}

/*!
 * The next surface to be encountered.
 */
CELER_FORCEINLINE_FUNCTION void
OrangeTrackView::next_surf(detail::OnLocalSurface const& s)
{
    states_.next_surf[track_slot_] = s.id();
    states_.next_sense[track_slot_] = s.unchecked_sense();
}

/*!
 * The level of the next surface to be encounted.
 */
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::next_surface_level(LevelId lev)
{
    states_.next_level[track_slot_] = lev;
}

//---------------------------------------------------------------------------//
// CONST STATE ACCESSORS
/*!
 * The current surface level.
 */
CELER_FORCEINLINE_FUNCTION LevelId const& OrangeTrackView::surface_level() const
{
    return states_.surface_level[track_slot_];
}

/*!
 * The local surface on the current surface level.
 */
CELER_FORCEINLINE_FUNCTION LocalSurfaceId const& OrangeTrackView::surf() const
{
    return states_.surf[track_slot_];
}

/*!
 * The sense on the current surface level.
 */
CELER_FORCEINLINE_FUNCTION Sense const& OrangeTrackView::sense() const
{
    return states_.sense[track_slot_];
}

/*!
 * The boundary on the current surface level.
 */
CELER_FORCEINLINE_FUNCTION BoundaryResult const&
OrangeTrackView::boundary() const
{
    return states_.boundary[track_slot_];
}

/*!
 * The next step distance.
 */
CELER_FORCEINLINE_FUNCTION real_type const& OrangeTrackView::next_step() const
{
    return states_.next_step[track_slot_];
}

/*!
 * The next surface to be encountered.
 */
CELER_FORCEINLINE_FUNCTION detail::OnLocalSurface
OrangeTrackView::next_surf() const
{
    return {states_.next_surf[track_slot_], states_.next_sense[track_slot_]};
}

/*!
 * The level of the next surface to be encounted.
 */
CELER_FORCEINLINE_FUNCTION LevelId const&
OrangeTrackView::next_surface_level() const
{
    return states_.next_level[track_slot_];
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Iterate over levels 1 to N to find the next step.
 *
 * Caller is responsible for finding the candidate next step on level 0, and
 * passing the resultant Intersection object as an argument.
 */
CELER_FUNCTION Propagation
OrangeTrackView::find_next_step_impl(detail::Intersection isect)
{
    TrackerVisitor visit_tracker{params_};

    // The level with minimum distance to intersection
    LevelId min_level{0};

    // Find the nearest intersection from level 0 to current level
    // inclusive, prefering the shallowest level (i.e., lowest univ_id)
    for (auto levelid : range(LevelId{1}, this->level() + 1))
    {
        auto univ_id = this->make_lsa(levelid).universe();
        auto local_isect = visit_tracker(
            [local_state = this->make_local_state(levelid), &isect](auto&& t) {
                return t.intersect(local_state, isect.distance);
            },
            univ_id);

        if (local_isect.distance < isect.distance)
        {
            isect = local_isect;
            min_level = levelid;
        }
    }

    this->next_step(isect.distance);
    this->next_surf(isect.surface);
    if (isect)
    {
        // Save level corresponding to the intersection
        this->next_surface_level(min_level);
    }

    Propagation result;
    result.distance = isect.distance;
    result.boundary = static_cast<bool>(isect);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the nearest boundary in any direction.
 *
 * The safety distance at a given point is the minimum safety distance over all
 * levels, since surface deduplication can potentionally elide bounding
 * surfaces at more deeply embedded levels.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety()
{
    CELER_EXPECT(!this->is_on_boundary());

    TrackerVisitor visit_tracker{params_};

    real_type min_safety_dist = numeric_limits<real_type>::infinity();

    for (auto lev : range(LevelId{this->level() + 1}))
    {
        auto lsa = this->make_lsa(lev);
        auto sd = visit_tracker(
            [&lsa](auto&& t) { return t.safety(lsa.pos(), lsa.vol()); },
            lsa.universe());
        min_safety_dist = celeritas::min(min_safety_dist, sd);
    }
    return min_safety_dist;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the nearest nearby boundary.
 *
 * Since we currently support only "simple" safety distances, we can't
 * eliminate anything by checking only nearby surfaces.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety(real_type)
{
    return this->find_safety();
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION Span<SenseValue> OrangeTrackView::make_temp_sense() const
{
    auto const max_faces = params_.scalars.max_faces;
    auto offset = track_slot_.get() * max_faces;
    return states_.temp_sense[AllItems<SenseValue, MemSpace::native>{}].subspan(
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
    if (level == this->surface_level())
    {
        local.surface = {this->surf(), this->sense()};
    }
    else
    {
        local.surface = {};
    }
    local.temp_sense = this->make_temp_sense();
    local.temp_next = this->make_temp_next();
    return local;
}

//---------------------------------------------------------------------------//
/*!
 * Whether any next step has been calculated.
 */
CELER_FORCEINLINE_FUNCTION bool OrangeTrackView::has_next_step() const
{
    return this->next_step() != 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the next intersecting surface has been found.
 */
CELER_FORCEINLINE_FUNCTION bool OrangeTrackView::has_next_surface() const
{
    return static_cast<bool>(states_.next_surf[track_slot_]);
}

//---------------------------------------------------------------------------//
/*!
 * Reset the next distance-to-boundary and surface.
 */
CELER_FUNCTION void OrangeTrackView::clear_next()
{
    this->next_step(0);
    states_.next_surf[track_slot_] = {};
    CELER_ENSURE(!this->has_next_step() && !this->has_next_surface());
}

//---------------------------------------------------------------------------//
/*!
 * Assign the surface on the current level.
 */
CELER_FUNCTION void
OrangeTrackView::surface(LevelId level, detail::OnLocalSurface surf)
{
    states_.surface_level[track_slot_] = level;
    states_.surf[track_slot_] = surf.id();
    states_.sense[track_slot_] = surf.unchecked_sense();
}

//---------------------------------------------------------------------------//
/*!
 * Clear the surface on the current level.
 */
CELER_FUNCTION void OrangeTrackView::clear_surface()
{
    states_.surface_level[track_slot_] = {};
    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Make a LevelStateAccessor for the current thread and level.
 */
CELER_FORCEINLINE_FUNCTION auto OrangeTrackView::make_lsa() const -> LSA
{
    return this->make_lsa(this->level());
}

//---------------------------------------------------------------------------//
/*!
 * Make a LevelStateAccessor for the current thread and a given level.
 */
CELER_FORCEINLINE_FUNCTION auto
OrangeTrackView::make_lsa(LevelId level) const -> LSA
{
    return LSA(&states_, track_slot_, level);
}

//---------------------------------------------------------------------------//
/*!
 * Get the daughter ID for the given volume in the given universe.
 *
 * \return DaughterId or {} if the current volume is a leaf.
 */
CELER_FUNCTION DaughterId OrangeTrackView::get_daughter(LSA const& lsa)
{
    TrackerVisitor visit_tracker{params_};
    return visit_tracker([&lsa](auto&& t) { return t.daughter(lsa.vol()); },
                         lsa.universe());
}

//---------------------------------------------------------------------------//
/*!
 * Get the transform ID for the given daughter.
 */
CELER_FUNCTION TransformId OrangeTrackView::get_transform(DaughterId daughter_id)
{
    CELER_EXPECT(daughter_id);
    return params_.daughters[daughter_id].trans_id;
}

//---------------------------------------------------------------------------//
/*!
 * Get the transform ID for the given daughter.
 */
CELER_FUNCTION TransformId OrangeTrackView::get_transform(LevelId lev)
{
    CELER_EXPECT(lev < this->level());
    LSA lsa(&states_, track_slot_, lev);
    return this->get_transform(this->get_daughter(lsa));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
