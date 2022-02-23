//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecgeomTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Config.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "base/ArrayUtils.hh"
#include "base/Macros.hh"
#include "base/NumericLimits.hh"
#include "base/SoftEqual.hh"
#include "detail/VecgeomCompatibility.hh"
#include "geometry/Types.hh"
#include "VecgeomData.hh"

#ifdef VECGEOM_USE_NAVINDEX
#    include "detail/BVHNavigator.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Operate on the device with shared (persistent) data and local state.
 *
 * \code
    VecgeomTrackView geom(vg_params_ref, vg_state_ref, thread_id);
   \endcode
 */
class VecgeomTrackView
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t = GeoTrackInitializer;
    using ParamsRef
        = VecgeomParamsData<Ownership::const_reference, MemSpace::native>;
    using StateRef = VecgeomStateData<Ownership::reference, MemSpace::native>;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        VecgeomTrackView& other; //!< Existing geometry
        Real3             dir;   //!< New direction
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION VecgeomTrackView(const ParamsRef& data,
                                           const StateRef&  stateview,
                                           ThreadId         id);

    // Initialize the state
    inline CELER_FUNCTION VecgeomTrackView&
    operator=(const Initializer_t& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION VecgeomTrackView&
    operator=(const DetailedInitializer& init);

    // Find the distance to the next boundary
    inline CELER_FUNCTION real_type find_next_step();

    // Find the safety at a given position within the current volume
    inline CELER_FUNCTION real_type find_safety(const Real3& pos);

    // Move to the boundary in preparation for crossing it
    inline CELER_FUNCTION void move_to_boundary();

    // Move within the volume
    inline CELER_FUNCTION void move_internal(real_type step);

    // Move within the volume to a specific point
    inline CELER_FUNCTION void move_internal(const Real3& pos);

    // Cross from one side of the current surface to the other
    inline CELER_FUNCTION void cross_boundary();

    //!@{
    //! State accessors
    CELER_FORCEINLINE_FUNCTION const Real3& pos() const { return pos_; }
    CELER_FORCEINLINE_FUNCTION const Real3& dir() const { return dir_; }
    //!@}

    // Change direction
    inline CELER_FUNCTION void set_dir(const Real3& newdir);

    // Get the volume ID in the current cell.
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;

    //! VecGeom states are never "on" a surface
    CELER_FUNCTION SurfaceId surface_id() const { return {}; }

    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static CELER_CONSTEXPR_FUNCTION real_type extra_push() { return 1e-13; }

  private:
    //// TYPES ////

    using Volume   = vecgeom::LogicalVolume;
    using NavState = vecgeom::NavigationState;

    //// DATA ////

    //! Shared/persistent geometry data
    const ParamsRef& params_;

    //!@{
    //! Referenced thread-local data
    NavState&  vgstate_;
    NavState&  vgnext_;
    Real3&     pos_;
    Real3&     dir_;
    real_type& next_step_;
    //!@}

    //// HELPER FUNCTIONS ////

    //! Whether the next distance-to-boundary has been found
    CELER_FUNCTION bool has_next_step() const { return next_step_ > 0; }

    //! Get a reference to the current volume
    inline CELER_FUNCTION const Volume& volume() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION
VecgeomTrackView::VecgeomTrackView(const ParamsRef& params,
                                   const StateRef&  states,
                                   ThreadId         thread)
    : params_(params)
    , vgstate_(states.vgstate.at(params_.max_depth, thread))
    , vgnext_(states.vgnext.at(params_.max_depth, thread))
    , pos_(states.pos[thread])
    , dir_(states.dir[thread])
    , next_step_(states.next_step[thread])
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
VecgeomTrackView::operator=(const Initializer_t& init)
{
    // Initialize position/direction
    pos_       = init.pos;
    dir_       = init.dir;
    next_step_ = 0;

    // Set up current state and locate daughter volume.
    vgstate_.Clear();
    const vecgeom::VPlacedVolume* worldvol       = params_.world_volume;
    const bool                    contains_point = true;

    // Note that LocateGlobalPoint sets `vgstate_`. If `vgstate_` is outside
    // (including possibly on the outside volume edge), the volume pointer it
    // returns would be null at this point.
#ifdef VECGEOM_USE_NAVINDEX
    detail::BVHNavigator::LocatePointIn(
#else
    vecgeom::GlobalLocator::LocateGlobalPoint(
#endif
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
VecgeomTrackView& VecgeomTrackView::operator=(const DetailedInitializer& init)
{
    if (this != &init.other)
    {
        // Copy the navigation state and position from the parent state
        init.other.vgstate_.CopyTo(&vgstate_);
        pos_ = init.other.pos_;
    }

    // Set up the next state and initialize the direction
    dir_       = init.dir;
    next_step_ = 0;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION real_type VecgeomTrackView::find_next_step()
{
    if (this->has_next_step())
    {
        // Next boundary distance is cached
        return next_step_;
    }
    else if (!this->is_outside())
    {
#ifdef VECGEOM_USE_NAVINDEX
        // Use BVH navigator to find internal distance
        // Note: AdePT provides max phys.length as maxStep
        //  - if used, next state = current state
        next_step_ = detail::BVHNavigator::ComputeStepAndNextVolume(
#else
        const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
        next_step_ = navigator->ComputeStepAndPropagatedState(
#endif
            detail::to_vector(pos_),
            detail::to_vector(dir_),
            vecgeom::kInfLength,
            vgstate_,
            vgnext_);
    }
    else
    {
        // Find distance to interior from outside world volume
        auto* pplvol = params_.world_volume;
        next_step_   = pplvol->DistanceToIn(detail::to_vector(pos_),
                                          detail::to_vector(dir_),
                                          vecgeom::kInfLength);

        vgnext_.Clear();
        if (next_step_ < vecgeom::kInfLength)
            vgnext_.Push(pplvol);
    }

    next_step_ = max(next_step_, this->extra_push());
    CELER_ENSURE(this->has_next_step());
    return next_step_;
}

//---------------------------------------------------------------------------//
/*!
 * Find the new safety at a given position within the current volume.
 */
CELER_FUNCTION real_type VecgeomTrackView::find_safety(const Real3& pos)
{
#ifdef VECGEOM_USE_NAVINDEX
    real_type safety = detail::BVHNavigator::ComputeSafety(
        detail::to_vector(pos), vgstate_);
#else
    const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
    real_type                  safety
        = navigator->GetSafetyEstimator()->ComputeSafetyForLocalPoint(
            detail::to_vector(pos), vgstate_.Top());
#endif
    return safety;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary but don't cross yet.
 */
CELER_FUNCTION void VecgeomTrackView::move_to_boundary()
{
    CELER_EXPECT(this->has_next_step());

    // Move next step
    axpy(next_step_, dir_, &pos_);
    next_step_ = 0.;
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
CELER_FUNCTION void VecgeomTrackView::cross_boundary()
{
    // Relocate to next tracking volume (maybe across multiple boundaries)
#ifdef VECGEOM_USE_NAVINDEX
    if (vgnext_.Top() != nullptr)
    {
        // BVH requires an lvalue temp_pos
        auto temp_pos = detail::to_vector(this->pos_);
        detail::BVHNavigator::RelocateToNextVolume(
            temp_pos, detail::to_vector(this->dir_), vgnext_);
    }
#endif

    vgstate_ = vgnext_;
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
    CELER_EXPECT(dist > 0 && dist < next_step_);

    // Move and update next_step_
    axpy(dist, dir_, &pos_);
    next_step_ -= dist;
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * \todo Currently it's up to the caller to make sure that the position is
 * "nearby". We should actually test this with a safety distance.
 */
CELER_FUNCTION void VecgeomTrackView::move_internal(const Real3& pos)
{
    pos_       = pos;
    next_step_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Change the track's direction.
 *
 * This happens after a scattering event or movement inside a magnetic field.
 * It resets the calculated distance-to-boundary.
 */
CELER_FUNCTION void VecgeomTrackView::set_dir(const Real3& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));
    dir_       = newdir;
    next_step_ = 0;
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
// PRIVATE MEMBER FUNCTIONS
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
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION const vecgeom::LogicalVolume& VecgeomTrackView::volume() const
{
    const vecgeom::VPlacedVolume* physvol_ptr = vgstate_.Top();
    CELER_ENSURE(physvol_ptr);
    return *physvol_ptr->GetLogicalVolume();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
