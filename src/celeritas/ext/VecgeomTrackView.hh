//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NavStateFwd.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/Types.hh"

#include "VecgeomData.hh"
#include "detail/BVHNavigator.hh"
#include "detail/VecgeomCompatibility.hh"

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
 */
class VecgeomTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    using ParamsRef = NativeCRef<VecgeomParamsData>;
    using StateRef = NativeRef<VecgeomStateData>;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        VecgeomTrackView& other;  //!< Existing geometry
        Real3 dir;  //!< New direction
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION VecgeomTrackView(ParamsRef const& data,
                                           StateRef const& stateview,
                                           TrackSlotId tid);

    // Initialize the state
    inline CELER_FUNCTION VecgeomTrackView&
    operator=(Initializer_t const& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION VecgeomTrackView&
    operator=(DetailedInitializer const& init);

    //// STATIC ACCESSORS ////

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static CELER_CONSTEXPR_FUNCTION real_type extra_push() { return 1e-13; }

    //// ACCESSORS ////

    //!@{
    //! State accessors
    CELER_FORCEINLINE_FUNCTION Real3 const& pos() const { return pos_; }
    CELER_FORCEINLINE_FUNCTION Real3 const& dir() const { return dir_; }
    //!@}

    // Get the volume ID in the current cell.
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;
    CELER_FORCEINLINE_FUNCTION int volume_physid() const;

    //!@{
    //! VecGeom states are never "on" a surface
    CELER_FUNCTION SurfaceId surface_id() const { return {}; }
    CELER_FUNCTION SurfaceId next_surface_id() const { return {}; }
    //!@}

    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    CELER_FORCEINLINE_FUNCTION bool is_on_boundary() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary (infinite max)
    inline CELER_FUNCTION Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the safety at the current position
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
    //// TYPES ////

    using Volume = vecgeom::LogicalVolume;
    using NavState = vecgeom::NavigationState;

    //// DATA ////

    //! Shared/persistent geometry data
    ParamsRef const& params_;

    //!@{
    //! Referenced thread-local data
    NavState& vgstate_;
    NavState& vgnext_;
    Real3& pos_;
    Real3& dir_;
    real_type& next_step_;
    //!@}

    //// HELPER FUNCTIONS ////

    // Whether any next distance-to-boundary has been found
    inline CELER_FUNCTION bool has_next_step() const;

    //! Get a reference to the current volume
    inline CELER_FUNCTION Volume const& volume() const;
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
    , vgstate_(states.vgstate.at(params_.max_depth, tid))
    , vgnext_(states.vgnext.at(params_.max_depth, tid))
    , pos_(states.pos[tid])
    , dir_(states.dir[tid])
    , next_step_(states.next_step[tid])
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 *
 * Expensive. This function should only be called to initialize an event from a
 * starting location and direction, but excess secondaries will also be
 * initialized this way.
 */
CELER_FUNCTION VecgeomTrackView&
VecgeomTrackView::operator=(Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    // Initialize position/direction
    pos_ = init.pos;
    dir_ = init.dir;
    next_step_ = 0;

    // Set up current state and locate daughter volume.
    vgstate_.Clear();
    vecgeom::VPlacedVolume const* worldvol = params_.world_volume;
    bool const contains_point = true;

    // Note that LocateGlobalPoint sets `vgstate_`. If `vgstate_` is outside
    // (including possibly on the outside volume edge), the volume pointer it
    // returns would be null at this point.
    detail::BVHNavigator::LocatePointIn(
        worldvol, detail::to_vector(pos_), vgstate_, contains_point);

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state from a direction and a copy of the parent state.
 *
 * This is a faster method of creating secondaries from a parent that has just
 * been absorbed, or when filling in an empty track from a parent that is still
 * alive.
 */
CELER_FUNCTION
VecgeomTrackView& VecgeomTrackView::operator=(DetailedInitializer const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    if (this != &init.other)
    {
        // Copy the navigation state and position from the parent state
        init.other.vgstate_.CopyTo(&vgstate_);
        pos_ = init.other.pos_;
    }

    // Set up the next state and initialize the direction
    dir_ = init.dir;
    next_step_ = 0;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FUNCTION VolumeId VecgeomTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return VolumeId{this->volume().id()};
}

//---------------------------------------------------------------------------//
/*!
 * Get the physical volume ID in the current cell.
 */
CELER_FUNCTION int VecgeomTrackView::volume_physid() const
{
    CELER_EXPECT(!this->is_outside());
    return this->vgstate_.Top()->id();
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
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation VecgeomTrackView::find_next_step()
{
    return this->find_next_step(vecgeom::kInfLength);
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation VecgeomTrackView::find_next_step(real_type max_step)
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
    else if (!vgnext_.IsOnBoundary() && next_step_ < max_step)
    {
        // Reset a previously found truncated distance
        next_step_ = 0;
    }

    if (this->has_next_step())
    {
        // Already cached
    }
    else if (!this->is_outside())
    {
        // Use BVH navigator to find internal distance
        next_step_ = detail::BVHNavigator::ComputeStepAndNextVolume(
            detail::to_vector(pos_),
            detail::to_vector(dir_),
            max_step,
            vgstate_,
            vgnext_);
        if (!vgnext_.IsOnBoundary())
        {
            // Soft equivalence between distance and max step is because the
            // BVH navigator subtracts and then re-adds a bump distance to the
            // step
            CELER_ASSERT(soft_equal(next_step_, max_step));
            next_step_ = max_step;
        }
        next_step_ = max(next_step_, this->extra_push());
    }
    else
    {
        // Find distance to interior from outside world volume
        auto* pplvol = params_.world_volume;
        next_step_ = pplvol->DistanceToIn(
            detail::to_vector(pos_), detail::to_vector(dir_), max_step);

        vgnext_.Clear();
        if (next_step_ <= max_step)
        {
            vgnext_.Push(pplvol);
            vgnext_.SetBoundaryState(true);
        }
        else
        {
            // Next step is valid but beyond the requested step
            Propagation result;
            result.distance = max_step;
            result.boundary = false;
            return result;
        }

        next_step_ = max(next_step_, this->extra_push());
    }

    Propagation result;
    result.distance = next_step_;
    result.boundary = vgnext_.IsOnBoundary();

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
    CELER_EXPECT(!this->is_outside());
    real_type safety = detail::BVHNavigator::ComputeSafety(
        detail::to_vector(this->pos()), vgstate_);

    // TODO: ComputeSafety can return negative safety distances: clamp it to
    // zero until we debug the underlying cause.
    return max<real_type>(safety, 0);
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary but don't cross yet.
 */
CELER_FUNCTION void VecgeomTrackView::move_to_boundary()
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(vgnext_.IsOnBoundary());

    // Move next step
    axpy(next_step_, dir_, &pos_);
    next_step_ = 0.;
    vgstate_.SetBoundaryState(true);  // XXX

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
    CELER_EXPECT(this->is_on_boundary());
    CELER_EXPECT(vgnext_.IsOnBoundary());

    // Relocate to next tracking volume (maybe across multiple boundaries)
    if (vgnext_.Top() != nullptr)
    {
        // BVH requires an lvalue temp_pos
        auto temp_pos = detail::to_vector(this->pos_);
        detail::BVHNavigator::RelocateToNextVolume(
            temp_pos, detail::to_vector(this->dir_), vgnext_);
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
    CELER_EXPECT(dist != next_step_ || !vgnext_.IsOnBoundary());

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
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION vecgeom::LogicalVolume const& VecgeomTrackView::volume() const
{
    vecgeom::VPlacedVolume const* physvol_ptr = vgstate_.Top();
    CELER_ENSURE(physvol_ptr);
    return *physvol_ptr->GetLogicalVolume();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
