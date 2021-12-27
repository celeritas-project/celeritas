//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.i.hh
//---------------------------------------------------------------------------//
#include <VecGeom/base/Config.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "base/ArrayUtils.hh"
#include "detail/VGCompatibility.hh"

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
GeoTrackView::GeoTrackView(const GeoParamsRef& data,
                           const GeoStateRef&  stateview,
                           const ThreadId&     thread)
    : shared_(data)
    , vgstate_(stateview.vgstate.at(shared_.max_depth, thread))
    , vgnext_(stateview.vgnext.at(shared_.max_depth, thread))
    , pos_(stateview.pos[thread])
    , dir_(stateview.dir[thread])
    , next_step_(stateview.next_step[thread])
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
CELER_FUNCTION GeoTrackView& GeoTrackView::operator=(const Initializer_t& init)
{
    // Initialize position/direction
    pos_       = init.pos;
    dir_       = init.dir;
    next_step_ = 0;

    // Set up current state and locate daughter volume.
    vgstate_.Clear();
    const vecgeom::VPlacedVolume* worldvol       = shared_.world_volume;
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
GeoTrackView& GeoTrackView::operator=(const DetailedInitializer& init)
{
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
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION real_type GeoTrackView::find_next_step()
{
    if (this->has_next_step())
    {
        // Next boundary distance is cached
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
        // Find distance to interior from inside world volume
        auto* pplvol = shared_.world_volume;
        next_step_   = pplvol->DistanceToIn(detail::to_vector(pos_),
                                          detail::to_vector(dir_),
                                          vecgeom::kInfLength);
        next_step_   = std::fmax(next_step_, this->extra_push());

        vgnext_.Clear();
        if (next_step_ < vecgeom::kInfLength)
            vgnext_.Push(pplvol);
    }

    CELER_ENSURE(this->has_next_step());
    return next_step_;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary and update volume accordingly.
 */
CELER_FUNCTION void GeoTrackView::move_across_boundary()
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
CELER_FUNCTION void GeoTrackView::move_internal(real_type dist)
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(dist > 0 && dist < next_step_);

    // Move and update next_step_
    axpy(dist, dir_, &pos_);
    next_step_ -= dist;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FUNCTION VolumeId GeoTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return VolumeId{this->volume().id()};
}

//---------------------------------------------------------------------------//
// PRIVATE CLASS FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION const vecgeom::LogicalVolume& GeoTrackView::volume() const
{
    const vecgeom::VPlacedVolume* physvol_ptr = vgstate_.Top();
    CELER_ENSURE(physvol_ptr);
    return *physvol_ptr->GetLogicalVolume();
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety to the closest geometric boundary.
 */
CELER_FUNCTION real_type GeoTrackView::find_safety(Real3 pos) const
{
#ifdef VECGEOM_USE_NAVINDEX
    return detail::BVHNavigator::ComputeSafety(
#else
    const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
    CELER_ASSERT(navigator);

    return navigator->GetSafetyEstimator()->ComputeSafety(
#endif
        detail::to_vector(pos), vgstate_);
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary from a given position
 * and to a direction and update the safety without updating the vegeom
 * state
 */
CELER_FUNCTION real_type GeoTrackView::compute_step(Real3      pos,
                                                    Real3      dir,
                                                    real_type* safety) const
{
    const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
    CELER_ASSERT(navigator);

    return navigator->ComputeStepAndSafety(detail::to_vector(pos),
                                           detail::to_vector(dir),
                                           vecgeom::kInfLength,
                                           vgstate_,
                                           true,
                                           *safety,
                                           false);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate to the next geometric boundary from a given position and
 * to a direction and update the vgstate
 */
CELER_FUNCTION void GeoTrackView::propagate_state(Real3 pos, Real3 dir) const
{
    const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
    CELER_ASSERT(navigator);

    navigator->ComputeStepAndPropagatedState(detail::to_vector(pos),
                                             detail::to_vector(dir),
                                             vecgeom::kInfLength,
                                             vgstate_,
                                             vgnext_);

    vgstate_ = vgnext_;
    vgstate_.SetBoundaryState(true);
    vgnext_.Clear();
}
} // namespace celeritas
