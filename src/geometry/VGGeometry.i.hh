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
    : data_(data), state_(stateview[id])
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 */
CELER_FUNCTION void VGGeometry::construct(const Real3& pos, const Real3& dir)
{
    // Initialize position/direction
    *state_.pos = pos;
    *state_.dir = dir;

    // Set up current state and locate daughter volume.
    state_.vgstate->Clear();
    const vecgeom::VPlacedVolume* volume         = data_.world_volume;
    const bool                    contains_point = true;

    // Note that LocateGlobalPoint sets state->vgstate.
    volume = vecgeom::GlobalLocator::LocateGlobalPoint(
        volume, detail::to_vector(pos), *state_.vgstate, contains_point);
    CHECK(volume);

    // Set up next state
    state_.vgnext->Clear();
    *state_.next_step = celeritas::numeric_limits<real_type>::quiet_NaN();
}

//---------------------------------------------------------------------------//
/*!
 * Clear the state.
 */
CELER_FUNCTION void VGGeometry::destroy()
{
    state_.vgstate->Clear();
    state_.vgnext->Clear();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION void VGGeometry::find_next_step()
{
    vecgeom::VNavigator const* navigator
        = this->volume().GetLogicalVolume()->GetNavigator();
    *state_.next_step = navigator->ComputeStepAndPropagatedState(
        detail::to_vector(*state_.pos),
        detail::to_vector(*state_.dir),
        vecgeom::kInfLength,
        *state_.vgstate,
        *state_.vgnext);
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
CELER_FUNCTION void VGGeometry::move_next_step()
{
    *state_.vgstate = *state_.vgnext;
    state_.vgnext->Clear();

    // Move the next step plus an extra fudge distance
    axpy(*state_.next_step + step_fudge(), *state_.dir, state_.pos);
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
    return state_.vgstate->IsOutside() ? Boundary::outside : Boundary::inside;
}

//---------------------------------------------------------------------------//
// PRIVATE CLASS FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume.
 */
CELER_FUNCTION const vecgeom::VPlacedVolume& VGGeometry::volume() const
{
    const vecgeom::VPlacedVolume* vol_ptr = state_.vgstate->Top();
    ENSURE(vol_ptr);
    return *vol_ptr;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
