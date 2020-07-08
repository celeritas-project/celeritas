//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGGeometry_hh
#define geometry_VGGeometry_hh

#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/navigation/NavigationState.h>

#include "base/NumericLimits.hh"
#include "VGStateView.hh"
#include "VGView.hh"
#include "Types.hh"
#include "detail/VGCompatibility.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Operate on the device with shared (persistent) data and local state.
 *
 * \code
    VGGeometry geom(vg_view, vg_state_view);
   \endcode
 */
class VGGeometry
{
  public:
    // Construct from persistent and state data
    CELER_INLINE_FUNCTION
    VGGeometry(const VGView&      data,
               const VGStateView& stateview,
               const ThreadId&    id);

    // Initialize the state
    CELER_INLINE_FUNCTION void construct(const Real3& pos, const Real3& dir);
    // Find the distance to the next boundary
    CELER_INLINE_FUNCTION void find_next_step();
    // Move to the next boundary
    CELER_INLINE_FUNCTION void move_next_step();
    // Destroy and invalidate the state
    CELER_INLINE_FUNCTION void destroy();

    //@{
    //! State accessors
    CELER_FUNCTION const Real3& pos() const { return pos_; }
    CELER_FUNCTION const Real3&    dir() const { return dir_; }
    CELER_FUNCTION real_type       next_step() const { return next_step_; }
    CELER_INLINE_FUNCTION VolumeId volume_id() const;
    CELER_INLINE_FUNCTION Boundary boundary() const;
    //@}

    // Fudge factor for movement (absolute distance)
    static CELER_CONSTEXPR_FUNCTION double step_fudge() { return 1e-6; }

  private:
    //@{
    //! Type aliases
    using Volume   = vecgeom::VPlacedVolume;
    using NavState = vecgeom::NavigationState;
    //@}

    //! Shared/persistent geometry data
    const VGView& shared_;

    //@{
    //! Referenced thread-local data
    NavState&  vgstate_;
    NavState&  vgnext_;
    Real3&     pos_;
    Real3&     dir_;
    real_type& next_step_;
    //@}

  private:
    // Get a reference to the state from a NavStatePool's pointer
    static CELER_INLINE_FUNCTION NavState&
                                 get_nav_state(void* state, int vgmaxdepth, ThreadId thread);

    // Get a reference to the current volume
    CELER_INLINE_FUNCTION const Volume& volume() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "VGGeometry.i.hh"

#endif // geometry_VGGeometry_hh
