//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsArrayCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/ParticleTrackView.hh"
#include "PhysicsArrayPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross sections or other physics data on device.
 *
 * \code
    PhysicsArrayCalculator calc_xs(xs_params);
    real_type xs = calc_xs(particle);
   \endcode
 */
class PhysicsArrayCalculator
{
  public:
    //@{
    //! Type aliases
    //@}

  public:
    // Construct from state-independent data
    explicit CELER_FUNCTION
    PhysicsArrayCalculator(const PhysicsArrayPointers& data)
        : data_(data)
    {
    }

    // Find and interpolate basesd on the particle track's current energy
    inline CELER_FUNCTION real_type
    operator()(const ParticleTrackView& particle) const;

  private:
    const PhysicsArrayPointers& data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsArrayCalculator.i.hh"
