//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Version.h>
// NOTE: must include Global before most other vecgeom/veccore includes
#include <VecGeom/base/Global.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "VecgeomData.hh"
#include "VecgeomTypes.hh"

#include "detail/VecgeomCompatibility.hh"

#if CELERITAS_VECGEOM_SURFACE
#    include "detail/SurfNavigator.hh"
#elif CELERITAS_VECGEOM_VERSION < 0x020000
#    include "detail/BVHNavigator.hh"
#else
#    include "detail/SolidsNavigator.hh"
#endif

#if CELER_VGNAV == CELER_VGNAV_PATH
#    include <VecGeom/navigation/NavStatePath.h>
#else
#    include "detail/VgNavStateWrapper.hh"
#endif

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "corecel/io/Repr.hh"
#    include "geocel/detail/LengthUnits.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Navigate through a VecGeom geometry on a single thread.
 *
 * For a description of ordering requirements, see:
 * \sa OrangeTrackView
 *
 * \code
    VecgeomTrackView geom(vg_params_ref, vg_state_ref, trackslot_id);
   \endcode
 *
 * The "next distance" is cached as part of `find_next_step`, but it is only
 * used when the immediate next call is `move_to_boundary`.
 */
class VecgeomTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    using ParamsRef = NativeCRef<VecgeomParamsData>;
    using StateRef = NativeRef<VecgeomStateData>;
#if CELERITAS_VECGEOM_SURFACE
    using Navigator = celeritas::detail::SurfNavigator;
#elif CELERITAS_VECGEOM_VERSION < 0x020000
    using Navigator = celeritas::detail::BVHNavigator;
#else
    using Navigator = celeritas::detail::SolidsNavigator;
#endif
    using ImplVolInstanceId = VgVolumeInstanceId;
    using real_type = vg_real_type;
    //!@}

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION VecgeomTrackView(ParamsRef const& data,
                                           StateRef const& stateview,
                                           TrackSlotId tid);

    // Initialize the state
    inline CELER_FUNCTION VecgeomTrackView&
    operator=(Initializer_t const& init);

    //// STATIC ACCESSORS ////

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static CELER_CONSTEXPR_FUNCTION real_type extra_push() { return 1e-13; }

    //// ACCESSORS ////

    //!@{
    //! State accessors
    CELER_FORCEINLINE_FUNCTION Real3 const& pos() const { return pos_; }
    CELER_FORCEINLINE_FUNCTION Real3 const& dir() const { return dir_; }
    //!@}

    // Get the canonical volume ID in the current impl volume
    inline CELER_FUNCTION VolumeId volume_id() const;
    // Get the ID of the current volume instance
    inline CELER_FUNCTION VolumeInstanceId volume_instance_id() const;
    // Get the depth in the geometry hierarchy
    inline CELER_FUNCTION VolumeLevelId volume_level() const;
    // Get the volume instance ID for all levels
    inline CELER_FUNCTION void
    volume_instance_id(Span<VolumeInstanceId> levels) const;

    // Get the current volume's ID
    inline CELER_FUNCTION ImplVolumeId impl_volume_id() const;
    // The current surface ID
    inline CELER_FUNCTION ImplSurfaceId impl_surface_id() const;
    // After 'find_next_step', the next straight-line surface
    inline CELER_FUNCTION ImplSurfaceId next_impl_surface_id() const;

    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    CELER_FORCEINLINE_FUNCTION bool is_on_boundary() const;
    //! Whether the last operation resulted in an error
    CELER_FORCEINLINE_FUNCTION bool failed() const { return failed_; }
    // Get the normal vector of the current surface
    inline CELER_FUNCTION Real3 normal() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary (infinite max)
    inline CELER_FUNCTION Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the safety at the current position (infinite max)
    inline CELER_FUNCTION real_type find_safety();

    // Find the safety at the current position up to a maximum step distance
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

    using VgLogVol = VgLogicalVolume<MemSpace::native>;
    using VgPlacedVol = VgPlacedVolume<MemSpace::native>;

#if CELER_VGNAV == CELER_VGNAV_PATH
    using NavStateWrapper = vecgeom::NavStatePath&;
#else
    using NavStateWrapper = detail::VgNavStateWrapper;
#endif

    //// DATA ////

    //! Shared/persistent geometry data
    ParamsRef const& params_;
    StateRef const& state_;
    TrackSlotId tid_;

    //!@{
    //! Referenced thread-local data
    NavStateWrapper vgstate_;
    NavStateWrapper vgnext_;
    Real3& pos_;
    Real3& dir_;
    VgSurfaceInt* next_surf_{nullptr};
    //!@}

    // Temporary data
    real_type next_step_{0};
    bool failed_{false};

    //// HELPER FUNCTIONS ////

    // Whether any next distance-to-boundary has been found
    inline CELER_FUNCTION bool has_next_step() const;

    // Whether the next distance-to-boundary is to a surface
    inline CELER_FUNCTION bool is_next_boundary() const;

    // Get a reference to the current volume instance
    inline CELER_FUNCTION VgPlacedVol const& physical_volume() const;

    // Get a reference to the current volume
    inline CELER_FUNCTION VgLogVol const& logical_volume() const;

    static CELER_CONSTEXPR_FUNCTION VgSurfaceInt null_surface()
    {
        return vg_null_surface;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION
VecgeomTrackView::VecgeomTrackView(ParamsRef const& params,
                                   StateRef const& states,
                                   TrackSlotId tid)
    : params_(params)
    , state_(states)
    , tid_(tid)
#if CELER_VGNAV == CELER_VGNAV_PATH
    // Nav path holds direct references to state with unused "last state"
    , vgstate_{states.state[tid]}
    , vgnext_{states.next_state[tid]}
#else
    , vgstate_{states.state[tid], states.boundary[tid]}
    , vgnext_{states.next_state[tid], states.next_boundary[tid]}
#endif
    , pos_(states.pos[tid])
    , dir_(states.dir[tid])
{
    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        next_surf_ = &states.next_surf[tid];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 *
 * If a valid parent ID is provided, the state is constructed from a direction
 * and a copy of the parent state.  This is a faster method of creating
 * secondaries from a parent that has just been absorbed, or when filling in an
 * empty track from a parent that is still alive.
 *
 * Otherwise, the state is initialized from a starting location and direction,
 * which is expensive.
 */
CELER_FUNCTION VecgeomTrackView&
VecgeomTrackView::operator=(Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));
    failed_ = false;

    // Initialize direction
    dir_ = init.dir;

    if (init.parent)
    {
        // Copy the navigation state and position from the parent state
        if (tid_ != init.parent)
        {
            VecgeomTrackView other(params_, state_, init.parent);
            other.vgstate_.CopyTo(&vgstate_);
            pos_ = other.pos_;
        }
        // Set up the next state and initialize the direction
        vgnext_ = vgstate_;

        CELER_ENSURE(this->pos() == init.pos);
        CELER_ENSURE(!this->has_next_step());
        return *this;
    }

    // Initialize the state from a position
    pos_ = init.pos;
    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        *next_surf_ = null_surface();
    }

    // Set up current state and locate daughter volume
    vgstate_.Clear();
#if CELERITAS_VECGEOM_SURFACE
    // VecGeom's BVHSurfNav takes `int pvol_id ` but vecgeom's navtuple/index
    // return NavIndex_t via `NavInd` :(
    VgPlacedVolumeInt world = vecgeom::NavigationState::WorldId();
#else
    auto const* world = params_.scalars.world<MemSpace::native>();
#endif
    // LocatePointIn sets `vgstate_`
    constexpr bool contains_point = true;
    Navigator::LocatePointIn(
        world, detail::to_vector(pos_), vgstate_, contains_point);

    if (CELER_UNLIKELY(vgstate_.IsOutside()))
    {
#if !CELER_DEVICE_COMPILE
        auto msg = CELER_LOG_LOCAL(error);
        msg << "Failed to initialize geometry state at " << repr(pos_) << ' '
            << lengthunits::native_label;
#endif
        failed_ = true;
    }

    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FORCEINLINE_FUNCTION VolumeId VecgeomTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(!params_.volumes.empty());

    return params_.volumes[this->impl_volume_id()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the physical volume ID in the current cell.
 *
 * If built with Geant4, this is the canonical volume instance ID. If built
 * with VGDML, this is an "implementation" instance ID.
 */
CELER_FUNCTION VolumeInstanceId VecgeomTrackView::volume_instance_id() const
{
    CELER_EXPECT(!this->is_outside());
    auto ipv_id = id_cast<ImplVolInstanceId>(this->physical_volume().id());
    return params_.volume_instances[ipv_id];
}

//---------------------------------------------------------------------------//
/*!
 * Get the depth in the geometry hierarchy.
 */
CELER_FUNCTION VolumeLevelId VecgeomTrackView::volume_level() const
{
    CELER_EXPECT(!this->is_outside());
    auto result = id_cast<VolumeLevelId>(vgstate_.GetLevel());
    CELER_ENSURE(result < params_.scalars.num_volume_levels);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance ID at each volume level.
 */
CELER_FUNCTION void
VecgeomTrackView::volume_instance_id(Span<VolumeInstanceId> levels) const
{
    CELER_EXPECT(id_cast<VolumeLevelId>(levels.size())
                 == this->volume_level() + 1);
    for (auto lev : range(levels.size()))
    {
        VgPlacedVol const* pv = vgstate_.At(lev);
        CELER_ASSERT(pv);
        auto ipv_id = id_cast<ImplVolInstanceId>(pv->id());
        levels[lev] = params_.volume_instances[ipv_id];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FORCEINLINE_FUNCTION ImplVolumeId VecgeomTrackView::impl_volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return id_cast<ImplVolumeId>(this->logical_volume().id());
}

//---------------------------------------------------------------------------//
/*!
 * The current surface frame ID.
 */
CELER_FUNCTION ImplSurfaceId VecgeomTrackView::impl_surface_id() const
{
    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        if (this->is_on_boundary())
        {
            return id_cast<ImplSurfaceId>(*next_surf_);
        }
    }
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * After 'find_next_step', the next straight-line surface.
 */
CELER_FUNCTION ImplSurfaceId VecgeomTrackView::next_impl_surface_id() const
{
    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        if (!this->is_on_boundary())
        {
            return id_cast<ImplSurfaceId>(*next_surf_);
        }
    }
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool VecgeomTrackView::is_outside() const
{
    return vgstate_.IsOutside();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is on the boundary of a volume.
 */
CELER_FUNCTION bool VecgeomTrackView::is_on_boundary() const
{
    return vgstate_.IsOnBoundary();
}

//---------------------------------------------------------------------------//
/*!
 * Get the surface normal of the boundary the track is currently on.
 */
CELER_FUNCTION Real3 VecgeomTrackView::normal() const
{
    // FIXME: temporarily return a bogus but valid surface normal
    return this->dir();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 *
 * \todo To support ray tracing from unknown distances from the world, we
 * could add another special function \c find_first_step .
 */
CELER_FUNCTION Propagation VecgeomTrackView::find_next_step()
{
    CELER_EXPECT(!this->is_outside());

    return this->find_next_step(vecgeom::kInfLength);
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation VecgeomTrackView::find_next_step(real_type max_step)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(max_step > 0);

    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        *next_surf_ = null_surface();
    }

    // TODO: vgnext is simply copied and the boundary flag optionally set
    next_step_ = Navigator::ComputeStepAndNextVolume(detail::to_vector(pos_),
                                                     detail::to_vector(dir_),
                                                     max_step,
                                                     vgstate_,
                                                     vgnext_
#if CELERITAS_VECGEOM_SURFACE
                                                     ,
                                                     *next_surf_
#endif
    );

    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        // Our accessor uses the next_surf_ state, but the temporary used for
        // vgnext_ should reflect the same result
        CELER_ASSERT((*next_surf_ != null_surface()) == vgnext_.IsOnBoundary());
    }

    next_step_ = max(next_step_, this->extra_push());

    if (!this->is_next_boundary())
    {
        // Soft equivalence between distance and max step is because the
        // BVH navigator subtracts and then re-adds a bump distance to the
        // step
        CELER_ASSERT(soft_equal(next_step_, max(max_step, this->extra_push())));
        next_step_ = max_step;
    }

    Propagation result;
    result.distance = next_step_;
    result.boundary = this->is_next_boundary();

    CELER_ENSURE(this->has_next_step());
    CELER_ENSURE(result.distance > 0);
    CELER_ENSURE(result.distance <= max(max_step, this->extra_push()));
    CELER_ENSURE(result.boundary || result.distance == max_step
                 || max_step < this->extra_push());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position.
 */
CELER_FUNCTION real_type VecgeomTrackView::find_safety()
{
    return this->find_safety(vecgeom::kInfLength);
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position up to a maximum distance.
 *
 * The safety within a step is only needed up to the end of the physics step
 * length.
 */
CELER_FUNCTION real_type VecgeomTrackView::find_safety(real_type max_radius)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(!this->is_on_boundary());
    CELER_EXPECT(max_radius > 0);

    real_type safety = Navigator::ComputeSafety(
        detail::to_vector(this->pos()), vgstate_, max_radius);
    safety = min<real_type>(safety, max_radius);

    // Since the reported "safety" is negative if we've moved slightly beyond
    // the boundary of a solid without crossing it, we must clamp to zero.
    return max<real_type>(safety, 0);
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary but don't cross yet.
 */
CELER_FUNCTION void VecgeomTrackView::move_to_boundary()
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(this->is_next_boundary());

    // Move next step
    axpy(next_step_, dir_, &pos_);
    next_step_ = 0;
    vgstate_.SetBoundaryState(true);

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
CELER_FUNCTION void VecgeomTrackView::cross_boundary()
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(this->is_on_boundary());
    CELER_EXPECT(this->is_next_boundary());

    // Relocate to next tracking volume (maybe across multiple boundaries)
    if (vgnext_.Top() != nullptr)
    {
        Navigator::RelocateToNextVolume(detail::to_vector(this->pos_),
                                        detail::to_vector(this->dir_),
#if CELERITAS_VECGEOM_SURFACE
                                        *next_surf_,
#endif
                                        vgnext_);
    }

    vgstate_ = vgnext_;

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume.
 *
 * The straight-line distance *must* be less than the distance to the
 * boundary.
 */
CELER_FUNCTION void VecgeomTrackView::move_internal(real_type dist)
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(dist > 0 && dist <= next_step_);
    CELER_EXPECT(dist != next_step_ || !this->is_next_boundary());

    // Move and update next_step_
    axpy(dist, dir_, &pos_);
    next_step_ -= dist;
    vgstate_.SetBoundaryState(false);

    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * \warning It's up to the caller to make sure that the position is
 * "nearby" and within the same volume.
 */
CELER_FUNCTION void VecgeomTrackView::move_internal(Real3 const& pos)
{
    pos_ = pos;
    next_step_ = 0;
    vgstate_.SetBoundaryState(false);

    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Change the track's direction.
 *
 * This happens after a scattering event or movement inside a magnetic field.
 * It resets the calculated distance-to-boundary.
 */
CELER_FUNCTION void VecgeomTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));
    dir_ = newdir;
    next_step_ = 0;
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether a next step has been calculated.
 */
CELER_FUNCTION bool VecgeomTrackView::has_next_step() const
{
    return next_step_ != 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the calculated next step will take track to next boundary.
 */
CELER_FUNCTION bool VecgeomTrackView::is_next_boundary() const
{
    CELER_EXPECT(this->has_next_step() || this->is_on_boundary());
    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        return *next_surf_ != null_surface();
    }
    else
    {
        return vgnext_.IsOnBoundary();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume.
 */
CELER_FUNCTION auto VecgeomTrackView::physical_volume() const
    -> VgPlacedVol const&
{
    VgPlacedVol const* physvol_ptr = vgstate_.Top();
    CELER_ENSURE(physvol_ptr);
    return *physvol_ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION auto VecgeomTrackView::logical_volume() const -> VgLogVol const&
{
    return *this->physical_volume().GetLogicalVolume();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
