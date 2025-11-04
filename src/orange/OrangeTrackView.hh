//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "LevelStateAccessor.hh"
#include "OrangeData.hh"
#include "OrangeTypes.hh"
#include "transform/TransformVisitor.hh"
#include "univ/SimpleUnitTracker.hh"
#include "univ/TrackerVisitor.hh"
#include "univ/UnivTypeTraits.hh"
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
 * The direction of \c normal is set to always point out of the volume the
 * track is currently in. On the boundary this is determined by the sense
 * of the track rather than its direction.
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
    using LSA = LevelStateAccessor;
    using UniverseIndexer = detail::UniverseIndexer;
    using real_type = ::celeritas::real_type;
    //!@}

  public:
    // Construct from params and state
    inline CELER_FUNCTION OrangeTrackView(ParamsRef const& params,
                                          StateRef const& states,
                                          TrackSlotId tid);

    // Initialize the state
    inline CELER_FUNCTION OrangeTrackView& operator=(Initializer_t const& init);

    //// STATE ACCESSORS ////

    // The current position
    inline CELER_FUNCTION Real3 const& pos() const;
    // The current direction
    inline CELER_FUNCTION Real3 const& dir() const;

    // Get the canonical volume ID in the current impl volume
    inline CELER_FUNCTION VolumeId volume_id() const;
    // Get the canonical volume instance ID in the current impl volume
    inline CELER_FUNCTION VolumeInstanceId volume_instance_id() const;
    // The level in the canonical volume graph
    inline CELER_FUNCTION VolumeLevelId volume_level() const;
    // Get the volume instance ID for all universe levels
    inline CELER_FUNCTION void volume_instance_id(Span<VolumeInstanceId>) const;

    // Whether the track is outside the valid geometry region
    inline CELER_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    inline CELER_FUNCTION bool is_on_boundary() const;
    //! Whether the last operation resulted in an error
    CELER_FORCEINLINE_FUNCTION bool failed() const { return failed_; }
    // Get the normal vector pointing out of the current volume
    inline CELER_FUNCTION Real3 normal() const;

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

    //// IMPLEMENTATION ACCESS ////

    // Geometry constant parameters
    inline CELER_FUNCTION OrangeParamsScalars const& scalars() const;

    // The level in the universe graph
    inline CELER_FUNCTION UnivLevelId univ_level() const;
    // The current implementation volume ID
    inline CELER_FUNCTION ImplVolumeId impl_volume_id() const;
    // The current surface ID
    inline CELER_FUNCTION ImplSurfaceId impl_surface_id() const;
    // After 'find_next_step', the next straight-line surface
    inline CELER_FUNCTION ImplSurfaceId next_impl_surface_id() const;

    // Make a universe indexer
    inline CELER_FUNCTION UniverseIndexer make_univ_indexer() const;
    // Make a LevelStateAccessor for the current thread and level
    inline CELER_FUNCTION LSA make_lsa() const;
    // Make a LevelStateAccessor for the current thread and a given level
    inline CELER_FUNCTION LSA make_lsa(UnivLevelId ulev_id) const;

  private:
    //// TYPES ////

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        TrackSlotId parent;  //!< Parent track with existing geometry
        Real3 const& dir;  //!< New direction
    };

    //// DATA ////

    ParamsRef const& params_;
    StateRef const& states_;
    TrackSlotId track_slot_;
    bool failed_{false};

    //// PRIVATE STATE MUTATORS ////

    inline CELER_FUNCTION void univ_level(UnivLevelId);
    inline CELER_FUNCTION void boundary(BoundaryResult);
    inline CELER_FUNCTION void next_step(real_type dist);
    inline CELER_FUNCTION void next_surf(detail::OnLocalSurface const&);
    inline CELER_FUNCTION void next_surface_level(UnivLevelId);

    //// PRIVATE STATE ACCESSORS ////

    inline CELER_FUNCTION UnivLevelId const& surface_univ_level() const;
    inline CELER_FUNCTION LocalSurfaceId const& surf() const;
    inline CELER_FUNCTION Sense const& sense() const;
    inline CELER_FUNCTION BoundaryResult const& boundary() const;
    inline CELER_FUNCTION real_type const& next_step() const;
    inline CELER_FUNCTION detail::OnLocalSurface next_surf() const;

    // The universe level of the next surface to be encountered
    inline CELER_FUNCTION UnivLevelId next_surface_univ_level() const;

    //// HELPER FUNCTIONS ////

    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION OrangeTrackView&
    operator=(DetailedInitializer const& init);

    // Iterate over universe levels to find the next step
    inline CELER_FUNCTION Propagation
    find_next_step_impl(detail::Intersection isect);

    // Create local sense reference
    inline CELER_FUNCTION Span<SenseValue> make_temp_sense() const;

    // Create local distance
    inline CELER_FUNCTION detail::TempNextFace make_temp_next() const;

    inline CELER_FUNCTION detail::LocalState
    make_local_state(UnivLevelId ulev_id) const;

    // Whether the next distance-to-boundary has been found
    inline CELER_FUNCTION bool has_next_step() const;

    // Whether the next surface has been found
    inline CELER_FUNCTION bool has_next_surface() const;

    // Invalidate the next distance-to-boundary and surface
    inline CELER_FUNCTION void clear_next();

    // Assign the surface at the current universe level
    inline CELER_FUNCTION void
    surface(UnivLevelId ulev_id, detail::OnLocalSurface surf);

    // Clear the surface at the current universe level
    inline CELER_FUNCTION void clear_surface();

    // Get the daughter ID for the volume in the universe (or null)
    inline CELER_FUNCTION DaughterId get_daughter(LSA const& lsa) const;

    // Get the transform ID for the given daughter.
    inline CELER_FUNCTION TransformId get_transform(DaughterId daughter_id) const;

    // Get the transform ID to increase the universe level by 1
    inline CELER_FUNCTION TransformId get_transform(UnivLevelId ulev_id) const;

    // The implementation volume ID at a given level
    inline CELER_FUNCTION ImplVolumeId impl_volume_id(UnivLevelId ulev_id) const;

    // Get the surface normal as defined by the geometry
    inline CELER_FUNCTION Real3 geo_normal() const;
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

    if (init.parent)
    {
        // Initialize from direction and copy of parent state
        *this = DetailedInitializer{init.parent, init.dir};
        CELER_ENSURE(this->pos() == init.pos);
        return *this;
    }

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
    UnivId univ_id = orange_global_univ;
    DaughterId daughter_id;
    UnivLevelId ulev_id{0};
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
            // but preserve the current universe level information
            failed_ = true;
            tinit.volume = orange_exterior_volume;
        }

        auto lsa = this->make_lsa(ulev_id);
        lsa.vol() = tinit.volume;
        lsa.pos() = local.pos;
        lsa.dir() = local.dir;
        lsa.univ() = univ_id;

        daughter_id = visit_tracker(
            [&tinit](auto&& t) { return t.daughter(tinit.volume); }, univ_id);

        if (daughter_id)
        {
            auto const& daughter = params_.daughters[daughter_id];
            // Apply "transform down" based on stored transform
            apply_transform(transform_down_local, daughter.trans_id);
            // Update universe and increase universe level
            univ_id = daughter.univ_id;
            ++ulev_id;
        }

    } while (daughter_id);

    // Save found universe level
    this->univ_level(ulev_id);

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

    if (track_slot_ != init.parent)
    {
        // Copy init track's position and logical state
        OrangeTrackView other(params_, states_, init.parent);
        this->univ_level(states_.univ_level[other.track_slot_]);
        this->surface(other.surface_univ_level(),
                      {other.surf(), other.sense()});
        this->boundary(other.boundary());

        for (auto ulev_id : range(this->univ_level() + 1))
        {
            // Copy all data accessed via LSA
            auto lsa = this->make_lsa(ulev_id);
            lsa = other.make_lsa(ulev_id);
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
    for (auto ulev_id : range(this->univ_level()))
    {
        auto lsa = this->make_lsa(ulev_id);
        lsa.dir() = localdir;
        apply_transform(rotate_down,
                        this->get_transform(this->get_daughter(lsa)));
    }

    // Save direction in deepest universe
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
    return this->make_lsa(orange_global_univ_level).pos();
}

//---------------------------------------------------------------------------//
/*!
 * The current direction.
 */
CELER_FUNCTION Real3 const& OrangeTrackView::dir() const
{
    return this->make_lsa(orange_global_univ_level).dir();
}

//---------------------------------------------------------------------------//
/*!
 * The current canonical volume ID.
 *
 * This is the volume identifier in the user's geometry model, not the ORANGE
 * implementation of it. For unit tests and certain use cases where the volumes
 * have not been loaded from Geant4 or a structured geometry model, it may not
 * be available.
 */
CELER_FUNCTION VolumeId OrangeTrackView::volume_id() const
{
    ImplVolumeId impl_id = this->impl_volume_id();
    // Return structural volume mapping
    CELER_ASSERT(impl_id);
    return params_.volume_ids[impl_id];
}

//---------------------------------------------------------------------------//
/*!
 * The current volume instance.
 */
CELER_FUNCTION VolumeInstanceId OrangeTrackView::volume_instance_id() const
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(!params_.volume_instance_ids.empty());

    // If we're in a 'background' volume, we don't know the PV until reaching
    // the parent placement (i.e., the volume instance in the parent universe)
    auto ui = this->make_univ_indexer();
    UnivLevelId ulev_id{this->univ_level()};
    auto get_vol_inst = [&]() {
        auto lsa = this->make_lsa(ulev_id);
        CELER_ASSERT(lsa.univ());
        ImplVolumeId impl_id = ui.global_volume(lsa.univ(), lsa.vol());
        return params_.volume_instance_ids[impl_id];
    };

    if (auto vi_id = get_vol_inst())
    {
        // Canonical mapping found in this universe: we're locally in a volume
        // placement
        return vi_id;
    }

    // Otherwise we're in a background volume, and the volume instance in the
    // parent universe *must* be a volume instance if this is a correctly
    // constructed geometry
    CELER_ASSERT(ulev_id != orange_global_univ_level);
    --ulev_id;
    return get_vol_inst();
}

//---------------------------------------------------------------------------//
/*!
 * The level in the canonical volume graph.
 */
CELER_FUNCTION VolumeLevelId OrangeTrackView::volume_level() const
{
    CELER_NOT_IMPLEMENTED("canonical level");
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance ID at every level.
 *
 * The input span size must be equal to the value of "level" plus one. The
 * top-most volume ("world" or level zero) starts at index zero, and child
 * volumes have higher level IDs. Note that Geant4 uses the \em reverse
 * nomenclature.
 *
 * \todo Implement \c parent_impl_volumes in OrangeData.
 */
CELER_FUNCTION void
OrangeTrackView::volume_instance_id(Span<VolumeInstanceId> levels) const
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(this->univ_level() < levels.size());

    // To guard against errors and enable unit tests, we first make sure we're
    // not going off the end. (If we are at the global level without correct
    // instance information, then this will just return a null ID.)
    CELER_NOT_IMPLEMENTED("canonical volume instance");
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool OrangeTrackView::is_outside() const
{
    // Zeroth volume in outermost universe is always the exterior by
    // construction in ORANGE
    auto lsa = this->make_lsa(orange_global_univ_level);
    return lsa.vol() == orange_exterior_volume;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is exactly on a surface.
 */
CELER_FORCEINLINE_FUNCTION bool OrangeTrackView::is_on_boundary() const
{
    return static_cast<bool>(this->surface_univ_level());
}

//---------------------------------------------------------------------------//
/*!
 * Get the normal vector of the current surface.
 *
 * The direction of the normal is determined by the sense of the track such
 * that the normal always points out of the volume that the track is currently
 * in.
 */
CELER_FUNCTION Real3 OrangeTrackView::normal() const
{
    CELER_EXPECT(this->is_on_boundary());

    auto normal = this->geo_normal();
    // Flip direction if on the outside of the surface
    if (this->sense() == Sense::outside)
    {
        normal = negate(normal);
    }

    return normal;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation OrangeTrackView::find_next_step()
{
    if (CELER_UNLIKELY(this->boundary() == BoundaryResult::entering))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }

    // Find intersection at the root level: always the first simple unit
    auto global_isect = [this] {
        SimpleUnitTracker t{params_, SimpleUnitId{0}};
        return t.intersect(this->make_local_state(orange_global_univ_level));
    }();
    // Find intersection for all deeper universe levels
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

    if (CELER_UNLIKELY(this->boundary() == BoundaryResult::entering))
    {
        // On a boundary, headed back in: next step is zero
        return {0, true};
    }

    // Find intersection at the root level: always the first simple unit
    auto global_isect = [this, &max_step] {
        SimpleUnitTracker t{params_, SimpleUnitId{0}};
        return t.intersect(this->make_local_state(orange_global_univ_level),
                           max_step);
    }();

    // Find intersection for all further levels
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
    CELER_EXPECT(this->boundary() != BoundaryResult::entering);
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(this->has_next_surface());

    // Physically move next step
    real_type const dist = this->next_step();
    for (auto ulev_id : range(this->univ_level() + 1))
    {
        auto lsa = this->make_lsa(ulev_id);
        axpy(dist, lsa.dir(), &lsa.pos());
    }

    this->boundary(BoundaryResult::entering);
    this->surface(this->next_surface_univ_level(), this->next_surf());
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
    for (auto i : range(this->univ_level() + 1))
    {
        auto lsa = this->make_lsa(UnivLevelId{i});
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
    // Transform all nonlocal universe levels
    auto local_pos = pos;
    auto apply_transform = TransformVisitor{params_};
    auto translate_down
        = [&local_pos](auto&& t) { local_pos = t.transform_down(local_pos); };
    for (auto ulev_id : range(this->univ_level()))
    {
        auto lsa = this->make_lsa(ulev_id);
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

    if (CELER_UNLIKELY(this->boundary() == BoundaryResult::exiting))
    {
        // Direction changed while on boundary leading to no change in
        // volume/surface. This is logically equivalent to a reflection.
        return;
    }

    // Cross surface by flipping the sense
    states_.sense[track_slot_] = flip_sense(this->sense());
    this->boundary(BoundaryResult::exiting);

    // Create local state from post-crossing level and updated sense
    UnivLevelId ulev_id{this->surface_univ_level()};
    UnivId univ;
    LocalVolumeId volume;
    detail::LocalState local;
    {
        auto lsa = this->make_lsa(ulev_id);
        univ = lsa.univ();
        local.pos = lsa.pos();
        local.dir = lsa.dir();
        local.volume = lsa.vol();
        local.surface = {this->surf(), this->sense()};
        local.temp_sense = this->make_temp_sense();
    }

    TrackerVisitor visit_tracker{params_};

    // Update the post-crossing volume by crossing the boundary of the "surface
    // crossing" level
    volume = visit_tracker(
        [&local](auto&& t) { return t.cross_boundary(local).volume; }, univ);
    if (CELER_UNLIKELY(!volume))
    {
        // Boundary crossing failure
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(error)
            << "track failed to cross local surface "
            << this->surf().unchecked_get() << " in universe "
            << univ.unchecked_get() << " at local position " << repr(local.pos)
            << " along local direction " << repr(local.dir);
#endif
        // Mark as failed and place in local "exterior" to end the search
        // but preserve the current level
        failed_ = true;
        volume = orange_exterior_volume;
    }
    make_lsa(ulev_id).vol() = volume;

    // Clear local surface before diving into daughters
    // TODO: this is where we'd do inter-universe mapping
    local.volume = {};
    local.surface = {};

    // Starting with the current level (i.e., next_surface_univ_level), iterate
    // down into the deepest level: *initializing* not *crossing*
    auto daughter_id = visit_tracker(
        [volume](auto&& t) { return t.daughter(volume); }, univ);
    while (daughter_id)
    {
        ++ulev_id;
        {
            // Update universe, local position/direction
            auto const& daughter = params_.daughters[daughter_id];
            TransformVisitor apply_transform{params_};
            auto transform_down_local = [&local](auto&& t) {
                local.pos = t.transform_down(local.pos);
                local.dir = t.rotate_down(local.dir);
            };
            apply_transform(transform_down_local, daughter.trans_id);
            univ = daughter.univ_id;
        }

        // Initialize in daughter and get IDs of volume and potential daughter
        volume = visit_tracker(
            [&local](auto&& t) { return t.initialize(local).volume; }, univ);

        if (!volume)
        {
#if !CELER_DEVICE_COMPILE
            auto msg = CELER_LOG_LOCAL(error);
            msg << "track failed to cross boundary: could not find associated "
                   "volume in universe "
                << univ.unchecked_get() << " at local position "
                << repr(local.pos);
#endif
            // Mark as failed and place in local "exterior" to end the search
            // but preserve the current level
            failed_ = true;
            volume = orange_exterior_volume;
        }
        daughter_id = visit_tracker(
            [volume](auto&& t) { return t.daughter(volume); }, univ);

        auto lsa = make_lsa(ulev_id);
        lsa.vol() = volume;
        lsa.pos() = local.pos;
        lsa.dir() = local.dir;
        lsa.univ() = univ;
    }

    // Save final univ_level
    this->univ_level(ulev_id);

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
CELER_FUNCTION void OrangeTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));

    if (this->is_on_boundary())
    {
        // Changing direction on a boundary, which may result in not leaving
        // current volume upon the cross_surface call
        auto normal = this->geo_normal();

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
    for (auto ulev_id : range(this->univ_level()))
    {
        auto lsa = this->make_lsa(ulev_id);
        lsa.dir() = localdir;
        apply_transform(rotate_down,
                        this->get_transform(this->get_daughter(lsa)));
    }
    // Save direction at deepest level
    this->make_lsa().dir() = localdir;

    this->clear_next();
}

//---------------------------------------------------------------------------//
// PUBLIC IMPLEMENTATION ACCESS
//---------------------------------------------------------------------------//
/*!
 * Geometry constant parameters.
 */
CELER_FUNCTION OrangeParamsScalars const& OrangeTrackView::scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
/*!
 * The track's level in the universe hierarchy.
 *
 * Zero corresponds to being in the global universe.
 */
CELER_FORCEINLINE_FUNCTION UnivLevelId OrangeTrackView::univ_level() const
{
    return states_.univ_level[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * The current "global" volume ID.
 *
 * \note It is allowable to call this function when "outside", because the
 * outside in ORANGE is just a special volume.
 */
CELER_FUNCTION ImplVolumeId OrangeTrackView::impl_volume_id() const
{
    auto lsa = this->make_lsa();
    return this->make_univ_indexer().global_volume(lsa.univ(), lsa.vol());
}

//---------------------------------------------------------------------------//
/*!
 * The current surface ID.
 */
CELER_FUNCTION ImplSurfaceId OrangeTrackView::impl_surface_id() const
{
    if (!this->is_on_boundary())
    {
        return {};
    }

    auto lsa = this->make_lsa(this->surface_univ_level());
    return this->make_univ_indexer().global_surface(lsa.univ(), this->surf());
}

//---------------------------------------------------------------------------//
/*!
 * After 'find_next_step', the next straight-line surface.
 */
CELER_FUNCTION ImplSurfaceId OrangeTrackView::next_impl_surface_id() const
{
    if (!this->has_next_surface())
    {
        return {};
    }

    auto lsa = this->make_lsa(this->next_surface_univ_level());
    return this->make_univ_indexer().global_surface(lsa.univ(),
                                                    this->next_surf().id());
}

//---------------------------------------------------------------------------//
/*!
 * Make a UniverseIndexer to convert local to global IDs.
 */
CELER_FORCEINLINE_FUNCTION auto OrangeTrackView::make_univ_indexer() const
    -> UniverseIndexer
{
    return UniverseIndexer{params_.univ_indexer_data};
}

//---------------------------------------------------------------------------//
/*!
 * Make a LevelStateAccessor for the current thread and level.
 *
 * Please treat as read-only outside this class!
 */
CELER_FORCEINLINE_FUNCTION auto OrangeTrackView::make_lsa() const -> LSA
{
    return this->make_lsa(this->univ_level());
}

//---------------------------------------------------------------------------//
/*!
 * Make a LevelStateAccessor for the current thread and a given univ_level.
 *
 * Note that access beyond the current level is allowable:
 * cross_boundary locally updates the univ_level before committing the
 * change.
 */
CELER_FORCEINLINE_FUNCTION auto
OrangeTrackView::make_lsa(UnivLevelId ulev_id) const -> LSA
{
    return LSA(params_.scalars, &states_, track_slot_, ulev_id);
}

//---------------------------------------------------------------------------//
// PRIVATE MUTABLE STATE ACCESSORS
//---------------------------------------------------------------------------//
//! The track's current universe level
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::univ_level(UnivLevelId ulev_id)
{
    states_.univ_level[track_slot_] = ulev_id;
}

//! The boundary on the current surface universe level
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::boundary(BoundaryResult br)
{
    states_.boundary[track_slot_] = br;
}

//! The next step distance
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::next_step(real_type dist)
{
    states_.next_step[track_slot_] = dist;
}

//! The next surface to be encountered
CELER_FORCEINLINE_FUNCTION void
OrangeTrackView::next_surf(detail::OnLocalSurface const& s)
{
    states_.next_surf[track_slot_] = s.id();
    states_.next_sense[track_slot_] = s.unchecked_sense();
}

//! The universe level of the next surface to be encountered
CELER_FORCEINLINE_FUNCTION void
OrangeTrackView::next_surface_level(UnivLevelId ulev_id)
{
    states_.next_univ_level[track_slot_] = ulev_id;
}

//---------------------------------------------------------------------------//
// PRIVATE CONST STATE ACCESSORS
//---------------------------------------------------------------------------//
//! The universe level of the current surface
CELER_FORCEINLINE_FUNCTION UnivLevelId const&
OrangeTrackView::surface_univ_level() const
{
    return states_.surface_univ_level[track_slot_];
}

//! The local surface on the current surface univ_level
CELER_FORCEINLINE_FUNCTION LocalSurfaceId const& OrangeTrackView::surf() const
{
    return states_.surf[track_slot_];
}

//! The sense on the current surface
CELER_FORCEINLINE_FUNCTION Sense const& OrangeTrackView::sense() const
{
    return states_.sense[track_slot_];
}

//! The boundary on the current surface
CELER_FORCEINLINE_FUNCTION BoundaryResult const&
OrangeTrackView::boundary() const
{
    return states_.boundary[track_slot_];
}

//! The next step distance
CELER_FORCEINLINE_FUNCTION real_type const& OrangeTrackView::next_step() const
{
    return states_.next_step[track_slot_];
}

//! The next surface to be encountered
CELER_FORCEINLINE_FUNCTION detail::OnLocalSurface
OrangeTrackView::next_surf() const
{
    return {states_.next_surf[track_slot_], states_.next_sense[track_slot_]};
}

//! The universe level of the next surface to be encountered
CELER_FORCEINLINE_FUNCTION UnivLevelId
OrangeTrackView::next_surface_univ_level() const
{
    return states_.next_univ_level[track_slot_];
}

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Iterate over universe levels 1 to N to find the next step.
 *
 * Caller is responsible for finding the candidate next step on level 0, and
 * passing the resultant Intersection object as an argument.
 */
CELER_FUNCTION Propagation
OrangeTrackView::find_next_step_impl(detail::Intersection isect)
{
    TrackerVisitor visit_tracker{params_};

    // The level with minimum distance to intersection
    UnivLevelId min_univ_level{0};

    // Find the nearest intersection from 0 to current
    // univ_level inclusive, preferring the shallowest univ_level
    // (i.e., lowest univ_id)
    for (auto ulev_id : range(UnivLevelId{1}, this->univ_level() + 1))
    {
        auto univ_id = this->make_lsa(ulev_id).univ();
        auto local_isect = visit_tracker(
            [local_state = this->make_local_state(ulev_id), &isect](auto&& t) {
                return t.intersect(local_state, isect.distance);
            },
            univ_id);

        if (local_isect.distance < isect.distance)
        {
            isect = local_isect;
            min_univ_level = ulev_id;
        }
    }

    this->next_step(isect.distance);
    this->next_surf(isect.surface);
    if (isect)
    {
        // Save univ_level corresponding to the intersection
        this->next_surface_level(min_univ_level);
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
 * universe levels, since surface deduplication can potentionally elide
 * bounding surfaces at more deeply embedded universe levels.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety()
{
    CELER_EXPECT(!this->is_on_boundary());

    TrackerVisitor visit_tracker{params_};

    real_type min_safety_dist = numeric_limits<real_type>::infinity();

    for (auto ulev_id : range(this->univ_level() + 1))
    {
        auto lsa = this->make_lsa(ulev_id);
        auto sd = visit_tracker(
            [&lsa](auto&& t) { return t.safety(lsa.pos(), lsa.vol()); },
            lsa.univ());
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
OrangeTrackView::make_local_state(UnivLevelId ulev_id) const
{
    detail::LocalState local;

    auto lsa = this->make_lsa(ulev_id);

    local.pos = lsa.pos();
    local.dir = lsa.dir();
    local.volume = lsa.vol();
    if (ulev_id == this->surface_univ_level())
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
 * Assign the surface on the current universe level.
 */
CELER_FUNCTION void
OrangeTrackView::surface(UnivLevelId ulev_id, detail::OnLocalSurface surf)
{
    states_.surface_univ_level[track_slot_] = ulev_id;
    states_.surf[track_slot_] = surf.id();
    states_.sense[track_slot_] = surf.unchecked_sense();
}

//---------------------------------------------------------------------------//
/*!
 * Clear the surface on the current universe level.
 */
CELER_FUNCTION void OrangeTrackView::clear_surface()
{
    states_.surface_univ_level[track_slot_] = {};
    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Get the daughter ID for the given volume in the given universe.
 *
 * \return DaughterId or {} if the current volume is a leaf.
 */
CELER_FUNCTION DaughterId OrangeTrackView::get_daughter(LSA const& lsa) const
{
    TrackerVisitor visit_tracker{params_};
    return visit_tracker([&lsa](auto&& t) { return t.daughter(lsa.vol()); },
                         lsa.univ());
}

//---------------------------------------------------------------------------//
/*!
 * Get the transform ID for the given daughter.
 */
CELER_FUNCTION TransformId
OrangeTrackView::get_transform(DaughterId daughter_id) const
{
    CELER_EXPECT(daughter_id);
    return params_.daughters[daughter_id].trans_id;
}

//---------------------------------------------------------------------------//
/*!
 * Get the transform ID for the given daughter.
 */
CELER_FUNCTION TransformId OrangeTrackView::get_transform(UnivLevelId ulev_id) const
{
    CELER_EXPECT(ulev_id < this->univ_level());
    LSA lsa(params_.scalars, &states_, track_slot_, ulev_id);
    return this->get_transform(this->get_daughter(lsa));
}

//---------------------------------------------------------------------------//
/*!
 * The global-indexed volume ID at a given univ level.
 *
 * \note It is allowable to call this function when "outside", because the
 * outside in ORANGE is just a special volume.
 */
CELER_FUNCTION ImplVolumeId
OrangeTrackView::impl_volume_id(UnivLevelId ulev_id) const
{
    CELER_EXPECT(ulev_id <= this->univ_level());
    auto lsa = this->make_lsa(ulev_id);
    auto ui = this->make_univ_indexer();
    return ui.global_volume(lsa.univ(), lsa.vol());
}

//---------------------------------------------------------------------------//
/*!
 * Get the normal vector of the current surface as defined by the geometry.
 */
CELER_FUNCTION Real3 OrangeTrackView::geo_normal() const
{
    CELER_EXPECT(this->is_on_boundary());

    auto normal = [this] {
        auto lsa = this->make_lsa(this->surface_univ_level());
        auto const& pos = lsa.pos();
        auto local_surf = this->surf();
        TrackerVisitor visit_tracker{params_};
        return visit_tracker(
            [&](auto&& t) { return t.normal(pos, local_surf); }, lsa.univ());
    }();

    // Rotate normal up to global coordinates
    auto apply_transform = TransformVisitor{params_};
    auto rotate_up = [&normal](auto&& t) { normal = t.rotate_up(normal); };
    for (auto ulev_id : range<int>(this->surface_univ_level().get()).step(-1))
    {
        apply_transform(rotate_up,
                        this->get_transform(id_cast<UnivLevelId>(ulev_id)));
    }

    CELER_ENSURE(is_soft_unit_vector(normal));
    return normal;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
