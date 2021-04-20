//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "base/Macros.hh"
#include "physics/base/Units.hh"
#include "field/FieldInterface.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"

#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/navigation/NavigationState.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read/write view to the field state of a single track.
 */

class FieldTrackView
{
  public:
    //@{
    //! Type aliases
    using NaviState = vecgeom::NavigationState;
    using Navigator = vecgeom::VNavigator;
    //@}

  public:
    // Construct from GeoTrackView and ParticleTrackView
    inline CELER_FUNCTION FieldTrackView(const GeoTrackView&      gt_view,
                                         const ParticleTrackView& pt_view);

    //@{
    //! Accessors
    CELER_FUNCTION bool on_boundary() const { return on_boundary_; }
    CELER_FUNCTION real_type step() const { return step_; }
    CELER_FUNCTION const OdeState& state() const { return state_; }
    CELER_FUNCTION real_type safety() const { return safety_; }
    CELER_FUNCTION Real3 origin() const { return origin_; }
    //@}

    //@{
    //! Setters
    CELER_FUNCTION void on_boundary(bool on_boundary)
    {
        on_boundary_ = on_boundary;
    }
    CELER_FUNCTION void step(real_type step)
    {
        CELER_EXPECT(step >= 0);
        step_ = step;
    }
    CELER_FUNCTION void state(const OdeState& state) { state_ = state; }

    CELER_FUNCTION void safety(real_type safety)
    {
        CELER_EXPECT(safety >= 0);
        safety_ = safety;
    }
    CELER_FUNCTION void origin(const Real3& origin) { origin_ = origin; }
    //@}

    /// STATIC PROPERTIES

    // Charge [elemental charge e+]
    CELER_FUNCTION units::ElementaryCharge charge() const { return charge_; };

    /// DERIVED PROPERTIES (indirection)

    // Linear propagation for a given step and update states and safety
    CELER_FUNCTION real_type linear_propagator(Real3     pos,
                                               Real3     dir,
                                               real_type step);

    // Update navigation and field states after a geometry limited step
    CELER_FUNCTION void update_vgstates();

  private:
    //@{
    //! thread-local data
    bool                    on_boundary_; //!< flag for a geometry limited step
    units::ElementaryCharge charge_;      //!< charge
    real_type               step_;        //!< step length
    OdeState                state_;       //!< position and momentum
    real_type               safety_;      //!< current safety
    Real3                   origin_;      //!< origin of safety
    NaviState&              vgstate_;     //!< vecgeom navigation state
    NaviState&              vgnext_;      //!< vecgeom next navigation state
    const Navigator*        navigator_;   //!< vecgeom navigator
    //@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldTrackView.i.hh"
