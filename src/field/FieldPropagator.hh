//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/Units.hh"

#include "FieldParamsPointers.hh"
#include "field/base/OdeArray.hh"
#include "field/base/FieldTrackView.hh"
#include "FieldIntegrator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 *  This is a high level interface of propagating a particle in a magnetic
 *  field.  It utilises an ODE solver (stepper) to track a charged particle,
 *  and drives based on an adaptive step contrel until the particle has
 *  traveled a set distance within a tolerance error or enters a new volume
 *  (intersect).
 *
 * \note This follows similar methods as in Geant4's G4PropagatorInField class.
 */
class FieldPropagator
{
    using ode_type  = OdeArray;
    using vec3_type = vecgeom::Vector3D<real_type>;

    // XXX TODO: use celetitas units
    // using LengthUnits = units::millimeter;
    // using Length = Quantity<LengthUnits>;

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION FieldPropagator(const FieldParamsPointers& shared,
                                          FieldIntegrator& integrator);

    // Propagation in a field
    inline CELER_FUNCTION real_type operator()(FieldTrackView& view);

    // >>> COMMON PROPERTIES

    //! Minimum acceptable step length for the propagation in a field
    static CELER_CONSTEXPR_FUNCTION real_type tolerance()
    {
        return 1e-5 * units::millimeter;
        //! XXX units::Millimeter{1e-5};
    }

  private:
    // Check whether there is a boundary crossing between two space points
    inline CELER_FUNCTION bool is_boundary_crossing(FieldTrackView& view,
                                                    const vec3_type x_start,
                                                    const vec3_type x_end,
                                                    vec3_type& intersect_point,
                                                    real_type& linear_step);

    // Find the first intersection point on a volume boundary
    inline CELER_FUNCTION bool find_intersect_point(FieldTrackView& view,
                                                    const ode_type& y_start,
                                                    ode_type&       y_end,
                                                    vec3_type& intersect_point,
                                                    real_type& hstep);

    // Helfer for find_intersect_point
    inline CELER_FUNCTION bool check_intersect(ode_type&  y_start,
                                               ode_type&  y_end,
                                               vec3_type& intersect_point,
                                               real_type& hstep);

  private:
    // Shared field parameters
    const FieldParamsPointers& shared_;

    // Field integrator
    FieldIntegrator& integrator_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldPropagator.i.hh"
