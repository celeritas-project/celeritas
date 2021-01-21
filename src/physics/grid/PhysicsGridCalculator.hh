//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Units.hh"
#include "base/Quantity.hh"
#include "XsGridPointers.hh"

namespace celeritas
{
class ParticleTrackView;

//---------------------------------------------------------------------------//
/*!
 * Find and interpolate physics data based on a track's energy.
 *
 * \todo Currently this is hard-coded to use "cross section grid pointers"
 * which have energy coordinates uniform in log space. The
 *
 * \code
    PhysicsGridCalculator calc_xs(xs_params);
    real_type xs = calc_xs(particle);
   \endcode
 */
class PhysicsGridCalculator
{
  public:
    //!@{
    //! Type aliases
    using Energy = Quantity<XsGridPointers::EnergyUnits>;
    //!@}

  public:
    // Construct from state-independent data
    explicit CELER_FUNCTION PhysicsGridCalculator(const XsGridPointers& data)
        : data_(data)
    {
    }

    // Find and interpolate based on the track state
    inline CELER_FUNCTION real_type
    operator()(const ParticleTrackView& particle) const;

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    const XsGridPointers& data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsGridCalculator.i.hh"
