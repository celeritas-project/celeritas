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
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a field.
 *
 * For a given initial state (position, momentum), it propagates a charged
 * particle along a curved trajectory up to an interaction length proposed by
 * a chosen physics process for the step, possibly integrating sub-steps by
 * an adaptive step control with a required accuracy of tracking in a
 * magnetic field and updates the final state (position, momentum) along with
 * the step actually taken.  If the final position is outside the current
 * volume, it returns a geometry limited step and the state at the
 * intersection between the curve trajectory and the first volume boundary
 * using an iterative step control method within a tolerance error imposed on
 * the closest distance between two positions by the field stepper and the
 * linear projection to the volume boundary.
 *
 * \note This follows similar methods as in Geant4's G4PropagatorInField class.
 */
template<class DriverT>
class FieldPropagator
{
  public:
    //! Output results
    struct result_type
    {
        real_type distance{0};     //!< Curved distance traveled
        bool      boundary{false}; //!< Hit a boundary before given distance
    };

  public:
    // Construct with shared parameters and the field driver
    inline CELER_FUNCTION FieldPropagator(const ParticleTrackView& particle,
                                          GeoTrackView*            track,
                                          DriverT*                 driver);

    // Move track to next volume boundary.
    inline CELER_FUNCTION result_type operator()();

    // Move track up to a user-provided distance, or to the next boundary
    inline CELER_FUNCTION result_type operator()(real_type dist);

  private:
    //// DATA ////

    GeoTrackView& track_;
    DriverT&      driver_;
    OdeState      state_;

    //// TYPES ////

    //! A helper input/output for private member functions
    struct Intersection
    {
        bool  intersected{false}; //!< Status of intersection
        Real3 pos{0, 0, 0};       //!< Intersection point on a volme boundary
        union
        {
            real_type step{0}; //!< Linear step length to the first boundary
            real_type scale;   //!< Scale for the next trial step length
        };
    };

    //// HELPER FUNCTIONS ////

    // Check whether the final state is crossed any boundary of volumes
    inline CELER_FUNCTION void query_intersection(const Real3&  beg_pos,
                                                  const Real3&  end_pos,
                                                  Intersection* intersect);

    // Find the intersection point if any boundary is crossed
    inline CELER_FUNCTION OdeState find_intersection(const OdeState& beg_state,
                                                     Intersection* intersect);
};
//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldPropagator.i.hh"
