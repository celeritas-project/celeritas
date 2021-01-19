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
    const vecgeom::VPlacedVolume* volume         = shared_.world_volume;
    const bool                    contains_point = true;

    // Note that LocateGlobalPoint sets vgstate. If `vgstate_` is outside
    // (including possibly on the outside volume edge), the volume pointer it
    // returns will be null
    vecgeom::GlobalLocator::LocateGlobalPoint(
        volume, detail::to_vector(pos_), vgstate_, contains_point);

    // Set up next state
    vgnext_.Clear();
    next_step_ = celeritas::numeric_limits<real_type>::quiet_NaN();
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
    vgnext_.Clear();
    next_step_ = celeritas::numeric_limits<real_type>::quiet_NaN();
    dir_       = init.dir;
    return *this;
}

//---------------------------------------------------------------------------//
//! Find the distance to the next geometric boundary.
CELER_FUNCTION void GeoTrackView::find_next_step()
{
    const vecgeom::LogicalVolume* logical_vol
        = this->volume().GetLogicalVolume();
    CELER_ASSERT(logical_vol);
    const vecgeom::VNavigator* navigator
        = this->volume().GetLogicalVolume()->GetNavigator();
    CELER_ASSERT(navigator);

    next_step_
        = navigator->ComputeStepAndPropagatedState(detail::to_vector(pos_),
                                                   detail::to_vector(dir_),
                                                   vecgeom::kInfLength,
                                                   vgstate_,
                                                   vgnext_);
}

//---------------------------------------------------------------------------//
//! Move to the next boundary and update volume accordingly
CELER_FUNCTION void GeoTrackView::move_next_step()
{
    // Move the next step plus an extra fudge distance
    axpy(next_step_, dir_, &pos_);
    this->move_next_volume();
}

//---------------------------------------------------------------------------//
//! Update state to next volume
CELER_FUNCTION void GeoTrackView::move_next_volume()
{
    vgstate_ = vgnext_;
    vgnext_.Clear();
}

//---------------------------------------------------------------------------//
//! Get the volume ID in the current cell.
CELER_FUNCTION VolumeId GeoTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return VolumeId{this->volume().id()};
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
//! Get a reference to the current volume.
CELER_FUNCTION const vecgeom::VPlacedVolume& GeoTrackView::volume() const
{
    const vecgeom::VPlacedVolume* vol_ptr = vgstate_.Top();
    CELER_ENSURE(vol_ptr);
    return *vol_ptr;
}

} // namespace celeritas
