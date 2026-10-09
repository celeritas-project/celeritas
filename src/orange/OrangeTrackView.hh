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
#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "OrangeData.hh"
#include "OrangeTypes.hh"
#include "transform/TransformVisitor.hh"
#include "univ/TrackerVisitor.hh"
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
    using UniverseIndexer = detail::UniverseIndexer;
    using real_type = ::celeritas::real_type;
    //!@}

  public:
    // Construct from params and state
    inline CELER_FUNCTION OrangeTrackView(
        ParamsRef const& params, StateRef const& states, TrackSlotId tid);

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
    // Visit every volume instance in the track's path, including world
    template<class F>
    inline CELER_FUNCTION void foreach_volume_path(F&& visit) const;

    // Geometry status
    inline CELER_FUNCTION GeoStatus geo_status() const;
    // Whether the track is outside the valid geometry region
    inline CELER_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    inline CELER_FUNCTION bool is_on_boundary() const;
    // Whether the last operation resulted in an error
    inline CELER_FUNCTION bool failed() const;
    // Get the normal vector pointing out of the current volume
    inline CELER_FUNCTION Real3 normal() const;

    //// OPERATIONS ////

    // DEPRECATED (remove in v0.8): infinite find_next_step
    inline CELER_FUNCTION Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // DEPRECATED (remove in v0.8): infinite find_safety
    inline CELER_FUNCTION real_type find_safety();

    // Find the distance to the nearest nearby boundary in any direction
    inline CELER_FUNCTION real_type find_safety(real_type max_step);

    // Move to the boundary in preparation for crossing it
    inline CELER_FUNCTION void move_to_boundary();

    // Forward compatibility: move to the boundary
    inline CELER_FUNCTION void move_to_boundary(real_type distance);

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

    // The current implementation volume ID
    inline CELER_FUNCTION ImplVolumeId impl_volume_id() const;
    // The current surface ID
    inline CELER_FUNCTION ImplSurfaceId impl_surface_id() const;
    // After 'find_next_step', the next straight-line surface
    inline CELER_FUNCTION ImplSurfaceId next_impl_surface_id() const;

    // Make a universe indexer
    inline CELER_FUNCTION UniverseIndexer make_univ_indexer() const;

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

    //// PRIVATE STATE MUTATORS ////

    inline CELER_FUNCTION void surface(detail::OnLocalSurface surf);
    inline CELER_FUNCTION void geo_status(GeoStatus);

    inline CELER_FUNCTION void next_step(real_type dist);
    inline CELER_FUNCTION void next_surf(detail::OnLocalSurface);

    //// PRIVATE STATE ACCESSORS ////

    inline CELER_FUNCTION UnivId univ() const;
    inline CELER_FUNCTION LocalVolumeId vol() const;
    inline CELER_FUNCTION LocalSurfaceId surf() const;
    inline CELER_FUNCTION Sense sense() const;

    inline CELER_FUNCTION real_type next_step() const;
    inline CELER_FUNCTION detail::OnLocalSurface next_surf() const;

    //// HELPER FUNCTIONS ////

    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION OrangeTrackView& operator=(
        DetailedInitializer const& init);

    // Iterate over universe levels to find the next step
    inline CELER_FUNCTION Propagation find_next_step_impl(
        detail::Intersection isect);

    // Create local distance
    inline CELER_FUNCTION detail::TempNextFace make_temp_next() const;

    inline CELER_FUNCTION detail::LocalState make_local_state() const;

    // Whether the next distance-to-boundary has been found
    inline CELER_FUNCTION bool has_next_step() const;

    // Whether the next surface has been found
    inline CELER_FUNCTION bool has_next_surface() const;

    // Invalidate the next distance-to-boundary and surface
    inline CELER_FUNCTION void clear_next();

    // Clear the surface at the current universe level
    inline CELER_FUNCTION void clear_surface();

    // Get the surface normal as defined by the geometry
    inline CELER_FUNCTION Real3 geo_normal() const;
};

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION OrangeTrackView::OrangeTrackView(
    ParamsRef const& params, StateRef const& states, TrackSlotId tid)
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
CELER_FUNCTION OrangeTrackView& OrangeTrackView::operator=(
    Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    if (init.parent)
    {
        // Initialize from direction and copy of parent state
        *this = DetailedInitializer{init.parent, init.dir};
        CELER_ENSURE(this->pos() == init.pos);
        return *this;
    }

    // Reset status and surface information
    this->clear_surface();
    this->clear_next();
    CELER_ASSERT(this->geo_status() == GeoStatus::interior);

    // Create local state
    detail::LocalState local;
    local.pos = init.pos;
    local.dir = init.dir;
    local.volume = {};
    local.surface = {};

    // Helpers for applying parent-to-daughter transformations
    TransformVisitor apply_transform{params_};
    auto transform_down_local = [&local](auto&& t) {
        local.pos = t.transform_down(local.pos);
        local.dir = t.rotate_down(local.dir);
    };

    // Recurse into daughter universes starting with the outermost universe
    UnivId univ_id = orange_global_univ;
    DaughterId daughter_id;
    do
    {
        TrackerVisitor visit_tracker{params_};
        auto tinit = visit_tracker(
            [&local](auto&& t) { return t.initialize(local); }, univ_id);

        if (CELER_UNLIKELY(!tinit.volume || tinit.surface))
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
            // Mark as failed and place in *local* "exterior" to end the search
            // and preserve diagnostic state
            this->geo_status(GeoStatus::error);
            tinit.volume = orange_exterior_volume;
        }

        states_.vol[track_slot_] = tinit.volume;
        states_.pos[track_slot_] = local.pos;
        states_.dir[track_slot_] = local.dir;
        states_.univ[track_slot_] = univ_id;

        daughter_id = visit_tracker(
            [&tinit](auto&& t) { return t.daughter(tinit.volume); }, univ_id);

        if (daughter_id)
        {
            CELER_ASSERT(this->geo_status() != GeoStatus::error);
            auto const& daughter = params_.daughters[daughter_id];
            // Apply "transform down" based on stored transform
            apply_transform(transform_down_local, daughter.trans_id);
            univ_id = daughter.univ_id;
        }
    } while (daughter_id);

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state from a direction and a copy of the parent state.
 */
CELER_FUNCTION OrangeTrackView& OrangeTrackView::operator=(
    DetailedInitializer const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    if (track_slot_ != init.parent)
    {
        // Copy init track's position and logical state
        OrangeTrackView other(params_, states_, init.parent);
        CELER_ASSERT(other.geo_status() != GeoStatus::error);
        this->surface({other.surf(), other.sense()});
        this->geo_status(other.geo_status());

        states_.pos[track_slot_] = other.pos();
        states_.vol[track_slot_] = other.vol();
        states_.univ[track_slot_] = other.univ();
    }

    // Clear the next step information since we're changing direction or
    // initializing a new state
    this->clear_next();

    // Save direction in deepest universe
    states_.dir[track_slot_] = init.dir;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * The current position.
 */
CELER_FUNCTION Real3 const& OrangeTrackView::pos() const
{
    return states_.pos[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * The current direction.
 */
CELER_FUNCTION Real3 const& OrangeTrackView::dir() const
{
    return states_.dir[track_slot_];
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

    return params_.volume_instance_ids[this->impl_volume_id()];
}

//---------------------------------------------------------------------------//
/*!
 * Apply the function with the volume instance ID and level.
 *
 * This can be used to construct a unique volume instance ID or fill a vector
 * with volume levels. The function for ORANGE is performed in local-to-global
 * order.
 */
template<class F>
CELER_FUNCTION void OrangeTrackView::foreach_volume_path(F&& visit) const
{
    CELER_EXPECT(!this->is_outside());

    VolumeLevelId next_vlev = this->volume_level() + 1;

    auto ui = this->make_univ_indexer();
    TrackerVisitor visit_tracker{params_};

    auto const univ = this->univ();

    // Initialize local volume from state
    LocalVolumeId lv_id = this->vol();
    // Loop over all local volumes that have local parents
    do
    {
        ImplVolumeId impl_id = ui.global_volume(univ, lv_id);
        if (auto vol_inst = params_.volume_instance_ids[impl_id].get())
        {
            CELER_ASSERT(next_vlev > VolumeLevelId{0});
            visit(--next_vlev, vol_inst);
            // Update to parent level
            lv_id = visit_tracker(
                [lv_id](auto&& t) { return t.local_parent(lv_id); }, univ);
        }
        else
        {
            // No volume instance at this level
            break;
        }
    } while (lv_id);
    CELER_ENSURE(next_vlev == VolumeLevelId{0});
}

//---------------------------------------------------------------------------//
/*!
 * The level in the canonical volume graph.
 */
CELER_FUNCTION VolumeLevelId OrangeTrackView::volume_level() const
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(!params_.volume_instance_ids.empty());

    TrackerVisitor visit_tracker{params_};

    vol_level_uint result = visit_tracker(
        [vol = this->vol()](auto&& t) { return t.local_vol_level(vol); },
        this->univ());

    return VolumeLevelId{result};
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance ID at every level.
 *
 * The input span size must be equal to the value of "level" plus one. The
 * top-most volume ("world" or level zero) starts at index zero, and child
 * volumes have higher level IDs. Note that Geant4 uses the \em reverse
 * nomenclature.
 */
CELER_FUNCTION void OrangeTrackView::volume_instance_id(
    Span<VolumeInstanceId> levels) const
{
    this->foreach_volume_path(
        [levels](VolumeLevelId lev, VolumeInstanceId vol_inst) {
            CELER_EXPECT(lev < levels.size());
            CELER_EXPECT(vol_inst);
            levels[*lev] = vol_inst;
        });
}

//---------------------------------------------------------------------------//
/*!
 * Geometry tracking state.
 */
CELER_FORCEINLINE_FUNCTION GeoStatus OrangeTrackView::geo_status() const
{
    return states_.status[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool OrangeTrackView::is_outside() const
{
    // Zeroth volume in outermost universe is always the exterior by
    // construction in ORANGE
    return this->univ() == orange_global_univ
           && this->vol() == orange_exterior_volume;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is exactly on a surface.
 */
CELER_FORCEINLINE_FUNCTION bool OrangeTrackView::is_on_boundary() const
{
    return static_cast<bool>(this->surf());
}

//---------------------------------------------------------------------------//
/*!
 * Whether the last operation resulted in an error.
 */
CELER_FORCEINLINE_FUNCTION bool OrangeTrackView::failed() const
{
    return this->geo_status() == GeoStatus::error;
}

//---------------------------------------------------------------------------//
/*!
 * Get the normal vector of the current surface.
 *
 * The direction of the normal is determined by the sense of the track such
 * that the normal always points out of the volume that the track is currently
 * in.
 * \todo This doesn't necessarily have the same meaning as in G4; we should
 * change so that the sign is arbitrary, and downstream use cases can flip
 * based on the geo status and its dot product with the direction.
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
 * Find a geometric boundary up to an infinite difference
 *
 * \deprecated Remove in v0.8: pass finite maximum step (precalculate from
 * world bbox if needed).
 */
[[deprecated]] CELER_FUNCTION Propagation OrangeTrackView::find_next_step()
{
    return this->find_next_step(numeric_limits<real_type>::infinity());
}

//---------------------------------------------------------------------------//
/*!
 * Find a nearby distance to the next geometric boundary up to a distance.
 *
 * Providing the "next step" (e.g., from the next collision point in a
 * physics-based simulation, or the image edge in a rasterization) may reduce
 * the number of surfaces needed to check, sort, or write to temporary memory,
 * thereby speeding up transport.
 *
 * \todo Prohibit when GeoStatus::boundary_inc
 */
CELER_FUNCTION Propagation OrangeTrackView::find_next_step(real_type next_step)
{
    CELER_EXPECT(next_step > 0);

    if (CELER_UNLIKELY(this->geo_status() == GeoStatus::boundary_inc))
    {
        // On a boundary, headed in: next step is zero
        return {0, true};
    }

    TrackerVisitor visit_tracker{params_};
    detail::OnLocalSurface next_local_surf{};

    // Find intersection for this local universe
    auto local_isect = visit_tracker(
        [local_state = this->make_local_state(), next_step](auto&& t) {
            return t.intersect(local_state, next_step);
        },
        this->univ());

    if (local_isect && local_isect.distance < next_step)
    {
        next_step = local_isect.distance;
        next_local_surf = local_isect.surface;
    }

    this->next_step(next_step);
    this->next_surf(next_local_surf);

    Propagation result;
    result.distance = next_step;
    result.boundary = static_cast<bool>(next_local_surf);

    CELER_ENSURE(result.distance <= next_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the nearest boundary in any direction.
 *
 * The safety distance at a given point is the minimum safety distance over all
 * universe levels, since surface deduplication can potentionally elide
 * bounding surfaces at more deeply embedded universe levels.
 *
 * \deprecated Remove in v0.8: use finite search instead
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety()
{
    return this->find_safety(numeric_limits<real_type>::infinity());
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the nearest nearby boundary.
 *
 * The safety distance at a given point is the minimum safety distance over all
 * universe levels, since surface deduplication can potentionally elide
 * bounding surfaces at more deeply embedded universe levels.
 *
 * Since we currently support only "simple" safety distances, we can't
 * eliminate anything by checking only nearby surfaces.
 */
CELER_FUNCTION real_type OrangeTrackView::find_safety(real_type)
{
    CELER_EXPECT(!this->is_on_boundary());

    TrackerVisitor visit_tracker{params_};
    // If we're intersecting a surface, the safety cannot be more than that.
    // Use that as a bound for degenerate cases such as starting in the exact
    // center of a sphere (where the safety can't correctly be calculated).
    // This fixes incorrectly large safety when a next step has been found,
    // necessary for CheckedGeoTrackView::find_next_step and more consistent in
    // general .
    real_type min_safety_dist = this->has_next_surface()
                                    ? this->next_step()
                                    : NumericLimits<real_type>::infinity();

    auto local_safety = visit_tracker(
        [this](auto&& t) { return t.safety(this->pos(), this->vol()); },
        this->univ());
    min_safety_dist = celeritas::min(min_safety_dist, local_safety);
    return min_safety_dist;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next straight-line boundary but do not change volume.
 *
 * Even though this does not change the universe or volume, it \em may change
 * the universe of the current surface.
 *
 * \deprecated Remove in v0.8: pass step length to boundary
 */
CELER_FUNCTION void OrangeTrackView::move_to_boundary()
{
    CELER_EXPECT(this->geo_status() != GeoStatus::boundary_inc);
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(this->has_next_surface());

    // Physically move next step
    real_type const dist = this->next_step();
    axpy(dist, this->dir(), &states_.pos[track_slot_]);

    this->geo_status(GeoStatus::boundary_inc);
    this->surface(this->next_surf());
    this->clear_next();

    CELER_ENSURE(this->geo_status() == GeoStatus::boundary_inc);
}

//---------------------------------------------------------------------------//
/*!
 * For forward compatibility with 0.8: step is managed independently.
 */
CELER_FUNCTION void OrangeTrackView::move_to_boundary(real_type step)
{
    CELER_EXPECT(step == this->next_step());
    this->move_to_boundary();
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
    CELER_EXPECT(this->geo_status() != GeoStatus::error);

    // Move and update the next step
    axpy(dist, this->dir(), &states_.pos[track_slot_]);
    this->next_step(this->next_step() - dist);
    this->clear_surface();

    CELER_ENSURE(this->geo_status() == GeoStatus::interior);
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
    CELER_EXPECT(this->geo_status() != GeoStatus::error);

    states_.pos[track_slot_] = pos;

    // Clear surface state and next-step info
    this->clear_surface();
    this->clear_next();

    CELER_ENSURE(this->geo_status() == GeoStatus::interior);
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary. This
 * should only be called once per boundary crossing.
 *
 * \todo Prohibit calling unless boundary_inc.
 */
CELER_FUNCTION void OrangeTrackView::cross_boundary()
{
    CELER_EXPECT(this->is_on_boundary());
    CELER_EXPECT(!this->has_next_step());

    if (this->geo_status() == GeoStatus::boundary_out)
    {
        // Direction changed while on boundary leading to no change in
        // volume/surface. This is logically equivalent to a reflection.
        return;
    }

    // Cross surface by flipping the sense
    states_.sense[track_slot_] = flip_sense(this->sense());
    this->geo_status(GeoStatus::boundary_out);

    // Create local state from post-crossing level and updated sense
    UnivId univ = this->univ();
    LocalVolumeId volume;
    detail::LocalState local;
    local.pos = this->pos();
    local.dir = this->dir();
    local.volume = this->vol();
    local.surface = {this->surf(), this->sense()};

    TrackerVisitor visit_tracker{params_};
    auto fail = [&] {
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(error)
            << "track failed to cross local surface "
            << this->surf().unchecked_get() << " in universe "
            << univ.unchecked_get() << " at local position " << repr(local.pos)
            << " along local direction " << repr(local.dir);
#endif
        // Mark as failed and place in *local* "exterior" to end the search and
        // preserve diagnostic state
        this->geo_status(GeoStatus::error);
        volume = orange_exterior_volume;
    };

    // Update the post-crossing volume by crossing the boundary of the "surface
    // crossing" level
    volume = visit_tracker(
        [&local](auto&& t) { return t.cross_boundary(local).volume; }, univ);
    if (CELER_UNLIKELY(!volume))
    {
        // Boundary crossing failure
        fail();
    }
    states_.vol[track_slot_] = volume;

    // Clear local surface before diving into daughters
    // TODO: this is where we'd do inter-universe mapping
    local.volume = {};
    local.surface = {};

    // Starting with the current level (i.e., next_univ_level), iterate
    // down into the deepest level: *initializing* not *crossing*
    auto daughter_id = visit_tracker(
        [volume](auto&& t) { return t.daughter(volume); }, univ);
    while (daughter_id)
    {
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

        if (CELER_UNLIKELY(!volume))
        {
            // Print message, change state, prepare to end loop
            fail();
            daughter_id = {};
        }
        daughter_id = visit_tracker(
            [volume](auto&& t) { return t.daughter(volume); }, univ);

        states_.vol[track_slot_] = volume;
        states_.pos[track_slot_] = local.pos;
        states_.dir[track_slot_] = local.dir;
        states_.univ[track_slot_] = univ;
    }

    CELER_ENSURE(this->geo_status() == GeoStatus::boundary_out
                 || this->geo_status() == GeoStatus::error);
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
 * \todo Remove use of geo_normal; instead, compare the local normal and local
 * direction versus the previous local direction at the lowest surface level.
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
        auto new_dot = dot_product(normal, newdir);
        if (CELER_UNLIKELY(new_dot == 0))
        {
#if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(error)
                << "track direction cannot change to " << newdir
                << " which is perpendicular to the current surface normal";
#endif
            // Scattered exactly perpendicular to the surface normal: oops!
            // The inc/out status now depends on the concavity at the local
            // point, or if we're along a planar surface then we can't move
            // consistently with the boundary state.
            this->geo_status(GeoStatus::error);
            return;
        }
        else if ((new_dot > 0) != (dot_product(normal, this->dir()) > 0))
        {
            // The boundary crossing direction has changed! Reverse our
            // plans to change the logical state and move to a new volume.
            this->geo_status(flip_boundary(this->geo_status()));
        }
    }

    // Save direction at deepest level
    states_.dir[track_slot_] = newdir;

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
 * The current "global" volume ID.
 *
 * \note It is allowable to call this function when "outside", because the
 * outside in ORANGE is just a special volume.
 */
CELER_FUNCTION ImplVolumeId OrangeTrackView::impl_volume_id() const
{
    return this->make_univ_indexer().global_volume(this->univ(), this->vol());
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

    return this->make_univ_indexer().global_surface(this->univ(), this->surf());
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

    return this->make_univ_indexer().global_surface(this->univ(),
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
// PRIVATE STATE MUTATORS
//---------------------------------------------------------------------------//
//! Assign the surface on the current universe level
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::surface(
    detail::OnLocalSurface surf)
{
    states_.surf[track_slot_] = surf.id();
    states_.sense[track_slot_] = surf.unchecked_sense();
}

//! Set the geo status
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::geo_status(GeoStatus gs)
{
    states_.status[track_slot_] = gs;
}

//! Set the next step distance
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::next_step(real_type dist)
{
    states_.next_step[track_slot_] = dist;
}

//! The next surface to be encountered
CELER_FORCEINLINE_FUNCTION void OrangeTrackView::next_surf(
    detail::OnLocalSurface s)
{
    states_.next_surf[track_slot_] = s.id();
    states_.next_sense[track_slot_] = s.unchecked_sense();
}

//---------------------------------------------------------------------------//
// PRIVATE STATE ACCESSORS
//---------------------------------------------------------------------------//
//! The current universe
CELER_FORCEINLINE_FUNCTION UnivId OrangeTrackView::univ() const
{
    return states_.univ[track_slot_];
}

//! The local volume in the current universe
CELER_FORCEINLINE_FUNCTION LocalVolumeId OrangeTrackView::vol() const
{
    return states_.vol[track_slot_];
}

//! The local surface on the current surface univ_level
CELER_FORCEINLINE_FUNCTION LocalSurfaceId OrangeTrackView::surf() const
{
    return states_.surf[track_slot_];
}

//! The sense on the current surface
CELER_FORCEINLINE_FUNCTION Sense OrangeTrackView::sense() const
{
    return states_.sense[track_slot_];
}

//! The next step distance
CELER_FORCEINLINE_FUNCTION real_type OrangeTrackView::next_step() const
{
    return states_.next_step[track_slot_];
}

//! The next surface to be encountered
CELER_FORCEINLINE_FUNCTION detail::OnLocalSurface
OrangeTrackView::next_surf() const
{
    return {states_.next_surf[track_slot_], states_.next_sense[track_slot_]};
}

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
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
CELER_FUNCTION detail::LocalState OrangeTrackView::make_local_state() const
{
    detail::LocalState local;

    local.pos = this->pos();
    local.dir = this->dir();
    local.volume = this->vol();
    local.surface = {this->surf(), this->sense()};
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
 * Clear the surface on the current universe level.
 *
 * \note If the previous track failed, then the error status will be cleared.
 * (This is necessary to initialize the geometry.)
 */
CELER_FUNCTION void OrangeTrackView::clear_surface()
{
    states_.surf[track_slot_] = {};
    this->geo_status(GeoStatus::interior);
    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Get the normal vector of the current surface as defined by the geometry.
 */
CELER_FUNCTION Real3 OrangeTrackView::geo_normal() const
{
    CELER_EXPECT(this->is_on_boundary());

    auto normal = [this] {
        auto const& pos = this->pos();
        auto local_surf = this->surf();
        TrackerVisitor visit_tracker{params_};
        return visit_tracker(
            [&](auto&& t) { return t.normal(pos, local_surf); }, this->univ());
    }();

    CELER_ENSURE(is_soft_unit_vector(normal));
    return normal;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
