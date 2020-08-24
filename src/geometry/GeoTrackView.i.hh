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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
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
    , mass_(stateview.mass[id.get()])
    , energy_(stateview.energy[id.get()])
    , momentum_(stateview.momentum[id.get()])
    , total_length_(stateview.total_length[id.get()])
    , proper_time_(stateview.proper_time[id.get()])
    , safety_(stateview.safety[id.get()])
    , step_(stateview.step[id.get()])
    , pstep_(stateview.pstep[id.get()])
    , snext_(stateview.snext[id.get()])
    , num_steps_(stateview.num_steps[id.get()])
    , status_(stateview.status[id.get()])
{
  mass_ = 9.109e-31 * units::kilogram;  // hard-coded e- mass for now
  //mass_ = constants::electron_mass_c2;
  energy_ = this->restEnergy();
}

CELER_FORCEINLINE_FUNCTION
void GeoTrackView::setEnergy(real_type energy)
{
  using namespace celeritas::units;
  using namespace celeritas::constants;
  double restE = this->restEnergy();
  energy_ = energy;
  momentum_ = sqrt(energy_ * energy_ - restE * restE) / constants::cLight;
}

CELER_FORCEINLINE_FUNCTION
void GeoTrackView::setKineticEnergy(real_type kinE)
{
  double restE = this->restEnergy();
  energy_ = restE + kinE;
  momentum_ = sqrt(energy_ * energy_ - restE * restE) / celeritas::constants::cLight;
}

CELER_FORCEINLINE_FUNCTION
void GeoTrackView::setMomentum(real_type p)
{
  momentum_ = p;
  double restE = this->restEnergy();
  double pc = p * celeritas::constants::cLight;
  energy_   = sqrt(restE * restE + pc * pc);
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
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION void GeoTrackView::find_next_step()
{
    vecgeom::VNavigator const* navigator
        = this->volume().GetLogicalVolume()->GetNavigator();
    next_step_
        = navigator->ComputeStepAndPropagatedState(detail::to_vector(pos_),
                                                   detail::to_vector(dir_),
                                                   vecgeom::kInfLength,
                                                   vgstate_,
                                                   vgnext_);
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
CELER_FUNCTION void GeoTrackView::move_next_step()
{
    vgstate_ = vgnext_;
    vgnext_.Clear();

    // Move the next step plus an extra fudge distance
    axpy(next_step_, dir_, &pos_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FUNCTION VolumeId GeoTrackView::volume_id() const
{
    //REQUIRE( this->boundary() == Boundary::No );
    return VolumeId{this->volume().id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return whether the track is inside or outside the valid geometry region.
 */
CELER_FUNCTION Boundary GeoTrackView::boundary() const
{
  return vgstate_.IsOnBoundary() ? Boundary::Yes : Boundary::No;
}

CELER_FUNCTION Boundary GeoTrackView::next_boundary() const
{
  return vgnext_.IsOnBoundary() ? Boundary::Yes : Boundary::No;
}

CELER_FUNCTION bool GeoTrackView::next_exiting()
{
  bool exiting = vgnext_.IsOutside();
  if ( exiting ) status_ = GeoTrackStatus::ExitingSetup;
  return exiting;
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
GeoTrackView::get_nav_state(void* state, CELER_MAYBE_UNUSED int vgmaxdepth, ThreadId thread)
    -> NavState&
{
    char* ptr = reinterpret_cast<char*>(state);
#ifdef __NVCC__
    ptr += vecgeom::cuda::NavigationState::SizeOfInstanceAlignAware(vgmaxdepth)
           * thread.get();
#else
    REQUIRE(thread.get() == 0);
#endif
    ENSURE(ptr);
    return *reinterpret_cast<NavState*>(ptr);
}

//---------------------------------------------------------------------------//
/*!
 * Check if current step has crossed a boundary, and update navState if necessary
 *
 * //When using the "cuda"-namespace navigation state (i.e., compiling with NVCC)
 * //it's necessary to transform the raw data pointer into an index.
 */
CELER_FUNCTION
bool GeoTrackView::has_same_path()
{
  //#### NOT USING YET THE NEW NAVIGATORS ####//
  // TODO: not using the direction yet here !!
  using namespace  vecgeom::GlobalLocator;
  bool samepath = HasSamePath(detail::to_vector(pos_), vgstate_, vgnext_);
  if (!samepath) {
    //tmpstate.CopyTo(track.fGeometryState.fNextpath);
#ifdef VECGEOM_CACHED_TRANS
    track.vgnext_->UpdateTopMatrix();
#endif
  }

  return samepath;
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume.
 */
CELER_FUNCTION const vecgeom::VPlacedVolume& GeoTrackView::volume() const
{
    const vecgeom::VPlacedVolume* vol_ptr = vgstate_.Top();
    ENSURE(vol_ptr);
    return *vol_ptr;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
