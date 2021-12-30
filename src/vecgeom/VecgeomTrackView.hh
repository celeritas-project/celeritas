//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecgeomTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/navigation/NavigationState.h>

#include "base/Macros.hh"
#include "base/NumericLimits.hh"
#include "geometry/Types.hh"
#include "VecgeomData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Operate on the device with shared (persistent) data and local state.
 *
 * \code
    VecgeomTrackView geom(vg_view, vg_state_view, thread_id);
   \endcode
 */
class VecgeomTrackView
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t = GeoTrackInitializer;
    using ParamsRef
        = VecgeomParamsData<Ownership::const_reference, MemSpace::native>;
    using StateRef = VecgeomStateData<Ownership::reference, MemSpace::native>;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        VecgeomTrackView& other; //!< Existing geometry
        Real3             dir;   //!< New direction
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION VecgeomTrackView(const ParamsRef& data,
                                           const StateRef&  stateview,
                                           ThreadId         id);

    // Initialize the state
    inline CELER_FUNCTION VecgeomTrackView&
                          operator=(const Initializer_t& init);
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION VecgeomTrackView&
                          operator=(const DetailedInitializer& init);

    // Find the distance to the next boundary
    inline CELER_FUNCTION real_type find_next_step();

    // Cross the next straight-line geometry boundary
    inline CELER_FUNCTION void move_across_boundary();

    // Move within the volume
    inline CELER_FUNCTION void move_internal(real_type step);

    // Move within the volume to a specific point
    inline CELER_FUNCTION void move_internal(const Real3& pos);

    //!@{
    //! State accessors
    CELER_FORCEINLINE_FUNCTION const Real3& pos() const { return pos_; }
    CELER_FORCEINLINE_FUNCTION const Real3& dir() const { return dir_; }
    //!@}

    // Change direction
    inline CELER_FUNCTION void set_dir(const Real3& newdir);

    // Get the volume ID in the current cell.
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;

    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static CELER_CONSTEXPR_FUNCTION real_type extra_push() { return 1e-13; }

  private:
    //// TYPES ////

    using Volume   = vecgeom::LogicalVolume;
    using NavState = vecgeom::NavigationState;

    //// DATA ////

    //! Shared/persistent geometry data
    const ParamsRef& shared_;

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
    CELER_FUNCTION bool has_next_step() const { return next_step_ > 0; }

    //! Get a reference to the current volume
    inline CELER_FUNCTION const Volume& volume() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "VecgeomTrackView.i.hh"
