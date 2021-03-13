//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.i.hh
//---------------------------------------------------------------------------//
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>
#include "base/ArrayUtils.hh"
#include "detail/VGCompatibility.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Construct from persistent and state data.
CELER_FUNCTION
GeoTrackView::GeoTrackView(const GeoParamsPointers& data,
                           const GeoStatePointers&  stateview,
                           const ThreadId&          id)
    : shared_(data)
    , vgstate_(GeoTrackView::get_nav_state(
          stateview.vgstate, stateview.vgmaxdepth, id))
    , vgnext_(GeoTrackView::get_nav_state(
          stateview.vgnext, stateview.vgmaxdepth, id))
    , pos_(stateview.pos[id.get()])
    , dir_(stateview.dir[id.get()])
    , next_step_(stateview.next_step[id.get()])
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

    // Note that LocateGlobalPoint sets vgstate. If `vgstate_` is outside
    // (including possibly on the outside volume edge), the volume pointer it
    // returns will be null
    vecgeom::GlobalLocator::LocateGlobalPoint(
        worldvol, detail::to_vector(pos_), vgstate_, contains_point);

    // Prepare for next step
    this->find_next_step(this->is_outside() ? worldvol : &this->volume());

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
CELER_FUNCTION void
GeoTrackView::find_next_step(vecgeom::VPlacedVolume const* pplvol)
{
    CELER_ASSERT(pplvol);

    const vecgeom::VNavigator* navigator
        = pplvol->GetLogicalVolume()->GetNavigator();
    CELER_ASSERT(navigator);

    if (this->is_outside())
    {
        // handling points outside of world volume
        const real_type large = vecgeom::kInfLength;
        next_step_            = pplvol->DistanceToIn(
            detail::to_vector(pos_), detail::to_vector(dir_), large);
        vgnext_.Clear();
        if (next_step_ < large)
        {
            vgnext_.Push(pplvol);
        }
    }
    else
    {
        next_step_
            = navigator->ComputeStepAndPropagatedState(detail::to_vector(pos_),
                                                       detail::to_vector(dir_),
                                                       vecgeom::kInfLength,
                                                       vgstate_,
                                                       vgnext_);
    }
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
//! Move to the next boundary and update volume accordingly
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
    this->find_next_step();
}

//---------------------------------------------------------------------------//
//! Get the volume ID in the current cell.
CELER_FUNCTION VolumeId GeoTrackView::volume_id() const
{
    CELER_EXPECT(!dirty_);
    return (this->is_outside() ? VolumeId{} : VolumeId{this->volume().id()});
}

//---------------------------------------------------------------------------//
// PRIVATE CLASS FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Determine the pointer to the navigation state for a particular index.
 *
 * When using the "cuda"-namespace navigation state (i.e., compiling with NVCC)
 * it's necessary to transform the raw data pointer into an index.
 */
CELER_FUNCTION auto
GeoTrackView::get_nav_state(void*                  state,
                            CELER_MAYBE_UNUSED int vgmaxdepth,
                            ThreadId               thread) -> NavState&
{
    CELER_EXPECT(state);
    char* ptr = reinterpret_cast<char*>(state);
#ifdef __NVCC__
    ptr += vecgeom::cuda::NavigationState::SizeOfInstanceAlignAware(vgmaxdepth)
           * thread.get();
#else
    CELER_EXPECT(thread.get() == 0);
#endif
    CELER_ENSURE(ptr);
    return *reinterpret_cast<NavState*>(ptr);
}

//---------------------------------------------------------------------------//
//! Get a reference to the current volume, or to world volume if outside
CELER_FUNCTION const vecgeom::VPlacedVolume& GeoTrackView::volume() const
{
    return this->is_outside() ? *(shared_.world_volume) : *this->vgstate_.Top();
}

} // namespace celeritas
