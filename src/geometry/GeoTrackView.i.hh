//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.i.hh
//---------------------------------------------------------------------------//
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include "base/ArrayUtils.hh"
#include "detail/VGCompatibility.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Construct from persistent and state data.
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
    , dirty_(true)
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
    pos_ = init.pos;
    dir_ = init.dir;

    // Set up current state and locate daughter volume.
    vgstate_.Clear();
    const vecgeom::VPlacedVolume* worldvol       = shared_.world_volume;
    const bool                    contains_point = true;

    // Note that LocateGlobalPoint sets `vgstate_`. If `vgstate_` is outside
    // (including possibly on the outside volume edge), the volume pointer it
    // returns would be null at this point.
    vecgeom::GlobalLocator::LocateGlobalPoint(
        worldvol, detail::to_vector(pos_), vgstate_, contains_point);

    // Prepare for next step. If outside, vgstate_ will be reset to return
    //  world volume instead of null.
    if (this->is_outside())
        this->find_next_step_outside();
    else
        this->find_next_step();

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

    this->find_next_step();
    return *this;
}

//---------------------------------------------------------------------------//
//! Find the distance to the next geometric boundary.
CELER_FUNCTION void GeoTrackView::find_next_step()
{
    const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
    CELER_ASSERT(navigator);

    next_step_
        = navigator->ComputeStepAndPropagatedState(detail::to_vector(pos_),
                                                   detail::to_vector(dir_),
                                                   vecgeom::kInfLength,
                                                   vgstate_,
                                                   vgnext_);
    dirty_ = false;
}

//---------------------------------------------------------------------------//
//! For outside points, find distance to world volume
CELER_FUNCTION void GeoTrackView::find_next_step_outside()
{
    CELER_EXPECT(this->is_outside());

    // handling points outside of world volume
    const vecgeom::VPlacedVolume* pplvol = shared_.world_volume;
    const real_type               large  = vecgeom::kInfLength;
    next_step_                           = pplvol->DistanceToIn(
        detail::to_vector(pos_), detail::to_vector(dir_), large);
    vgnext_.Clear();
    if (next_step_ < large)
        vgnext_.Push(pplvol);

    dirty_ = false;
}

//---------------------------------------------------------------------------//
//! Move to the next boundary and update volume accordingly
CELER_FUNCTION real_type GeoTrackView::move_to_boundary()
{
    if (dirty_)
        this->find_next_step();

    // Move the next step plus an extra fudge distance
    real_type dist = next_step_ + this->extra_push();
    axpy(dist, dir_, &pos_);
    next_step_ = 0.;
    this->move_next_volume();
    return dist;
}

//---------------------------------------------------------------------------//
//! Move to the next boundary and update volume accordingly
CELER_FUNCTION real_type GeoTrackView::move_next_step()
{
    if (dirty_)
        this->find_next_step();
    real_type dist = next_step_;
    axpy(next_step_, dir_, &pos_);
    next_step_ = 0.;
    this->move_next_volume();
    return dist;
}

//---------------------------------------------------------------------------//
//! Move by a given distance. If a boundary to be crossed, stop there instead
CELER_FUNCTION real_type GeoTrackView::move_by(real_type dist)
{
    CELER_EXPECT(dist > 0.);

    if (dirty_)
        this->find_next_step();

    // do not move beyond next boundary!
    if (dist >= next_step_)
        return this->move_to_boundary();

    // move and update next_step_
    axpy(dist, dir_, &pos_);
    next_step_ -= dist;
    return dist;
}

//---------------------------------------------------------------------------//
//! Update state to next volume
CELER_FUNCTION void GeoTrackView::move_next_volume()
{
    vgstate_ = vgnext_;
    if (this->is_outside())
        this->find_next_step_outside();
    else
        this->find_next_step();
}

//---------------------------------------------------------------------------//
//! Get the volume ID in the current cell.
CELER_FUNCTION VolumeId GeoTrackView::volume_id() const
{
    CELER_EXPECT(!dirty_);
    return (this->is_outside() ? VolumeId{}
                               : VolumeId{(unsigned int)this->volume().id()});
}

//---------------------------------------------------------------------------//
// PRIVATE CLASS FUNCTIONS
//---------------------------------------------------------------------------//
//! Get a reference to the current volume, or to world volume if outside
CELER_FUNCTION const vecgeom::LogicalVolume& GeoTrackView::volume() const
{
    const vecgeom::VPlacedVolume* physvol_ptr = vgstate_.Top();
    CELER_ENSURE(physvol_ptr);
    return *physvol_ptr->GetLogicalVolume();
}

//---------------------------------------------------------------------------//
// HELPER METHODS
//---------------------------------------------------------------------------//
//! Find the safety to the closest geometric boundary.
CELER_FUNCTION real_type GeoTrackView::find_safety(Real3 pos) const
{
    const vecgeom::VNavigator* navigator = this->volume().GetNavigator();
    CELER_ASSERT(navigator);

    return navigator->GetSafetyEstimator()->ComputeSafety(
        detail::to_vector(pos), vgstate_);
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary from a given position and
 * to a direction and update the safety without updating the vegeom state
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
