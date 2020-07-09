//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.i.hh
//---------------------------------------------------------------------------//

#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION
VGGeometry::VGGeometry(const VGView&      data,
                       const VGStateView& stateview,
                       const ThreadId&    id)
    : shared_(data)
    , vgstate_(VGGeometry::get_nav_state(
          stateview.vgstate, stateview.vgmaxdepth, id))
    , vgnext_(VGGeometry::get_nav_state(
          stateview.vgnext, stateview.vgmaxdepth, id))
    , pos_(stateview.pos[id.get()])
    , dir_(stateview.dir[id.get()])
    , next_step_(stateview.next_step[id.get()])
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 */
CELER_FUNCTION void VGGeometry::construct(const Real3& pos, const Real3& dir)
{
    // Initialize position/direction
    pos_ = pos;
    dir_ = dir;
    next_step_ = celeritas::numeric_limits<real_type>::quiet_NaN();
}

//---------------------------------------------------------------------------//
/*!
 * Clear the state.
 */
CELER_FUNCTION void VGGeometry::destroy()
{
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION void VGGeometry::find_next_step()
{
    next_step_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
CELER_FUNCTION void VGGeometry::move_next_step()
{
    // Move the next step plus an extra fudge distance
    axpy(next_step_ + step_fudge(), dir_, &pos_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FUNCTION VolumeId VGGeometry::volume_id() const
{
    return VolumeId{this->volume().id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return whether the track is inside or outside the valid geometry region.
 */
CELER_FUNCTION Boundary VGGeometry::boundary() const
{
    return Boundary::outside;
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
VGGeometry::get_nav_state(void* state, int vgmaxdepth, ThreadId thread)
    -> NavState&
{
    char* ptr = reinterpret_cast<char*>(state);
#ifdef __NVCC__
    ptr += vecgeom::cuda::NavigationState::SizeOf(vgmaxdepth) * thread.get();
#else
    REQUIRE(thread.get() == 0);
#endif
    ENSURE(ptr);
    return *reinterpret_cast<NavState*>(ptr);
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume.
 */
CELER_FUNCTION const vecgeom::VPlacedVolume& VGGeometry::volume() const
{
    const vecgeom::VPlacedVolume* vol_ptr = nullptr;
    // const vecgeom::VPlacedVolume* vol_ptr = vgstate_.Top();
    ENSURE(vol_ptr);
    return *vol_ptr;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
