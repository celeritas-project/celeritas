//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecgeomTrackView.i.hh
//---------------------------------------------------------------------------//
#include <VecGeom/base/Config.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "base/ArrayUtils.hh"
#include "base/SoftEqual.hh"
#include "detail/VecgeomCompatibility.hh"

#ifdef VECGEOM_USE_NAVINDEX
#    include "detail/BVHNavigator.hh"
#endif

namespace celeritas
{
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
 * starting location and direction. Secondaries will initialize their states
 * from a copy of the parent.
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

    next_step_ = std::fmax(next_step_, this->extra_push());
    CELER_ENSURE(this->has_next_step());
    return next_step_;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary and update volume accordingly.
 */
CELER_FUNCTION void VecgeomTrackView::move_across_boundary()
{
    CELER_EXPECT(this->has_next_step());

    // Move next step
    axpy(next_step_, dir_, &pos_);
    next_step_ = 0.;

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

    vgstate_ = vgnext_; // BVH relocation requires this extra step
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
