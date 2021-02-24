//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OdeArray.hh"

#include "base/Types.hh"
#include "base/Macros.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/base/ParticleTrackView.hh"

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/navigation/NavigationState.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read/write view to the field state of a single track.
 *
 * These functions are used in classes for the magnetic field.
 */

class FieldTrackView
{
  public:
    //@{
    //! Type aliases
    using ode_type  = OdeArray;
    using vec3_type = vecgeom::Vector3D<real_type>;
    using NaviState = vecgeom::NavigationState;
    using Navigator = vecgeom::VNavigator;
    //@}

  public:
    // Construct from GeoTrackView and ParticleTrackView
    inline CELER_FUNCTION
    FieldTrackView(const GeoTrackView&      gt_view,
                   const ParticleTrackView& pt_view);


    // Copy Construct
    inline CELER_FUNCTION 
    FieldTrackView(const FieldTrackView& other) = default;

    //@{
    //! Accessors
    CELER_FUNCTION real_type h() const { return h_; }
    CELER_FUNCTION const ode_type& y() const { return y_; }
    CELER_FUNCTION real_type safety() const { return safety_; }
    CELER_FUNCTION vec3_type origin() const { return origin_; }
    //@}

    //@{
    //! Modifiers via non-const references
    CELER_FUNCTION real_type& h() { return h_; }
    CELER_FUNCTION ode_type&  y() { return y_; }
    CELER_FUNCTION real_type& safety() { return safety_; }
    CELER_FUNCTION vec3_type& origin() { return origin_; }
    //@}

    /// STATIC PROPERTIES

    // Rest mass [MeV/c^2]
    CELER_FUNCTION units::MevMass mass() const { return m_; } ;

    // Charge [elemental charge e+]
    CELER_FUNCTION units::ElementaryCharge charge() const { return q_; };

    /// DERIVED PROPERTIES (indirection)

    // momentun magnitude square [MeV/c]^2
    inline CELER_FUNCTION real_type momentum_squre() const;

    // Linear propagation for a given step and update states and safety
    CELER_FUNCTION real_type linear_propagator(vec3_type pos,
                                               vec3_type dir,
                                               real_type step);

    // Update navigation and field states after a geometry limited step  
    CELER_FUNCTION void update_vgstates();

  private:
    //@{
    //! thread-local data
    units::MevMass                 m_; //!< mass
    units::ElementaryCharge        q_; //!< charge
    real_type                      h_; //!< step length
    ode_type                       y_; //!< position and momentum
    real_type                 safety_; //!< current safety
    vec3_type                 origin_; //!< origin of safety
    NaviState&               vgstate_; //!< vecgeom navigation state
    NaviState&                vgnext_; //!< vecgeom next navigation state
    const Navigator*       navigator_; //!< vecgeom navigator
    //@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldTrackView.i.hh"
