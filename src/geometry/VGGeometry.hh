//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGGeometry_hh
#define geometry_VGGeometry_hh

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
 * Operate on the device with persistent data and local state.
 *
 * \code
    VGGeometry geom(vg_view, vg_state_view);
   \endcode
 */
class VGGeometry
{
  public:
    //@{
    //! Type aliases
    //@}

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
    CELER_FUNCTION const Real3& pos() const { return *state_.pos; }
    CELER_FUNCTION const Real3& dir() const { return *state_.dir; }
    CELER_FUNCTION real_type    next_step() const { return *state_.next_step; }
    CELER_INLINE_FUNCTION VolumeId volume_id() const;
    CELER_INLINE_FUNCTION Boundary boundary() const;
    //@}

    // Fudge factor for movement (absolute distance)
    static CELER_CONSTEXPR_FUNCTION double step_fudge() { return 1e-6; }

  private:
    // Get a reference to the current volume
    CELER_INLINE_FUNCTION const vecgeom::VPlacedVolume& volume() const;

  private:
    const VGView&    data_;
    VGStateView::Ref state_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "VGGeometry.i.hh"

#endif // geometry_VGGeometry_hh
