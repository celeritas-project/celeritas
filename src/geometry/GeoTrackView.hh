//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/navigation/NavigationState.h>

#include "base/Macros.hh"
#include "base/NumericLimits.hh"
#include "GeoData.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Operate on the device with shared (persistent) data and local state.
 *
 * \code
    GeoTrackView geom(vg_view, vg_state_view, thread_id);
   \endcode
 */
class GeoTrackView
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t = GeoTrackInitializer;
    using GeoParamsRef
        = GeoParamsData<Ownership::const_reference, MemSpace::native>;
    using GeoStateRef = GeoStateData<Ownership::reference, MemSpace::native>;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        GeoTrackView& other; //!< Existing geometry
        Real3         dir;   //!< New direction
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION GeoTrackView(const GeoParamsRef& data,
                                       const GeoStateRef&  stateview,
                                       const ThreadId&     id);

    // Initialize the state
    inline CELER_FUNCTION GeoTrackView& operator=(const Initializer_t& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION GeoTrackView&
                          operator=(const DetailedInitializer& init);

    // Find the distance to the next boundary
    inline CELER_FUNCTION real_type find_next_step();

    // Cross the next straight-line geometry boundary
    inline CELER_FUNCTION void move_across_boundary();

    // Move within the volume
    inline CELER_FUNCTION void move_internal(real_type step);

    //!@{
    //! State accessors
    CELER_FUNCTION const Real3& pos() const { return pos_; }
    CELER_FUNCTION const Real3& dir() const { return dir_; }
    //!@}

    //!@{
    //! State modifiers will force state update before next step
    CELER_FUNCTION void set_pos(const Real3& newpos)
    {
        pos_       = newpos;
        next_step_ = 0;
    }
    CELER_FUNCTION void set_dir(const Real3& newdir)
    {
        dir_       = newdir;
        next_step_ = 0;
    }
    //!@}

    //! Get the volume ID in the current cell.
    inline CELER_FUNCTION VolumeId volume_id() const;

    //! Whether the track is inside or outside the valid geometry region
    CELER_FUNCTION bool is_outside() const { return vgstate_.IsOutside(); }

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static CELER_CONSTEXPR_FUNCTION real_type extra_push() { return 1e-13; }

  private:
    //// TYPES ////

    using Volume   = vecgeom::LogicalVolume;
    using NavState = vecgeom::NavigationState;

    //// DATA ////

    //! Shared/persistent geometry data
    const GeoParamsRef& shared_;

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
    inline CELER_FUNCTION bool has_next_step() const { return next_step_ > 0; }

  public:
    //! Get a reference to the current volume
    inline CELER_FUNCTION const Volume& volume() const;

    //!@{
    //! Helper methods
    // Find the safety to the closest geometric boundary
    inline CELER_FUNCTION real_type find_safety(Real3 pos) const;

    // Find the distance to the next geometric boundary and update the safety
    inline CELER_FUNCTION real_type compute_step(Real3      pos,
                                                 Real3      dir,
                                                 real_type* safety) const;

    // Propagate to the next volume and update the vgstate
    inline CELER_FUNCTION void propagate_state(Real3 pos, Real3 dir) const;
    //!@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "GeoTrackView.i.hh"
