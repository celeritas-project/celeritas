//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/navigation/NavigationState.h>

#include "base/Macros.hh"
#include "base/NumericLimits.hh"
#include "GeoInterface.hh"
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
    using Initializer_t = GeoStateInitializer;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        GeoTrackView& other; //!< Existing geometry
        Real3         dir;   //!< New direction
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION GeoTrackView(const GeoParamsPointers& data,
                                       const GeoStatePointers&  stateview,
                                       const ThreadId&          id);

    // Initialize the state
    inline CELER_FUNCTION GeoTrackView& operator=(const Initializer_t& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION GeoTrackView&
                          operator=(const DetailedInitializer& init);
    // Find the distance to the next boundary
    inline CELER_FUNCTION void find_next_step();
    // Move to the next boundary
    inline CELER_FUNCTION void move_next_step();

    // Update current volume, called whenever move reaches boundary
    inline CELER_FUNCTION void move_next_volume();

    //!@{
    //! State accessors
    CELER_FUNCTION const Real3& pos() const { return pos_; }
    CELER_FUNCTION const Real3& dir() const { return dir_; }
    CELER_FUNCTION real_type    next_step() const { return next_step_; }
    //!@}

    //!@{
    //! State modifiers via non-const references
    CELER_FUNCTION Real3& pos() { return pos_; }
    CELER_FUNCTION Real3& dir() { return dir_; }
    CELER_FUNCTION real_type& next_step() { return next_step_; }
    //!@}

    //! Get the volume ID in the current cell.
    inline CELER_FUNCTION VolumeId volume_id() const;

    //! Whether the track is inside or outside the valid geometry region
    CELER_FUNCTION bool is_outside() const { return vgstate_.IsOutside(); }

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static CELER_CONSTEXPR_FUNCTION real_type tolerance() { return 1e-13; }

  private:
    //!@{
    //! Type aliases
    using Volume   = vecgeom::VPlacedVolume;
    using NavState = vecgeom::NavigationState;
    //!@}

    //! Shared/persistent geometry data
    const GeoParamsPointers& shared_;

    //!@{
    //! Referenced thread-local data
    NavState&  vgstate_;
    NavState&  vgnext_;
    Real3&     pos_;
    Real3&     dir_;
    real_type& next_step_;
    //!@}

  private:
    //! Get a reference to the state from a NavStatePool's pointer
    static inline CELER_FUNCTION NavState&
    get_nav_state(void* state, int vgmaxdepth, ThreadId thread);

  public:
    //! Get a reference to the current volume
    inline CELER_FUNCTION const Volume& volume() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "GeoTrackView.i.hh"
