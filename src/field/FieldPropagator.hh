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

#include "field/FieldTrackView.hh"
#include "field/FieldParamsPointers.hh"
#include "field/FieldInterface.hh"
#include "field/FieldDriver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 *  This is a high level interface of propagating a particle in a magnetic
 *  field.  It utilises a magnetic field driver based on an adaptive step
 *  control and an ODE solver (stepper) to track a charged particle until
 *  it travels along a curved trajectory for a given step length within a
 *  required accuracy or intersects with a new volume (geometry limited step).
 *
 * \note This follows similar methods as in Geant4's G4PropagatorInField class.
 */
class FieldPropagator
{
  public:
    // Construct with shared parameters and the field driver
    inline CELER_FUNCTION
    FieldPropagator(const FieldParamsPointers& shared, FieldDriver& driver);

    // Propagation in a field
    inline CELER_FUNCTION real_type operator()(FieldTrackView* view);

  private:
    // A helper input/output for private member functions
    struct IntersectionIO
    {
        bool  status{false}; //!< Status of intersection
        Real3 pos{0, 0, 0};  //!< Intersection point on a volme boundary
        union
        {
            real_type step{0}; //!< Linear step length to the first boundary
            real_type scale;   //!< Scale for the next trial step length
        };
    };

    // Check whether the final state is crossed any boundary of volumes
    inline CELER_FUNCTION void check_intersection(FieldTrackView* view,
                                                  const Real3     beg_pos,
                                                  const Real3     end_pos,
                                                  IntersectionIO* intersect);

    // Find the intersection point if any boundary is crossed
    inline CELER_FUNCTION OdeState
                          locate_intersection(FieldTrackView* view,
                                              const OdeState  beg_state,
                                              IntersectionIO* intersect);

    // >>> COMMON PROPERTIES

    // A scale adjustment factor when IntersectionIO::scale is unity
    static CELER_CONSTEXPR_FUNCTION real_type scale_factor() { return 1.01; }

  private:
    // Shared field parameters
    const FieldParamsPointers& shared_;

    // Field driver
    FieldDriver& driver_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldPropagator.i.hh"
