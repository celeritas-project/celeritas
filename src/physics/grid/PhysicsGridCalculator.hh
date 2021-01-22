//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Quantity.hh"
#include "XsGridPointers.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Find and interpolate physics data based on a track's energy.
 *
 * \todo Currently this is hard-coded to use "cross section grid pointers"
 * which have energy coordinates uniform in log space. This should
 * be expanded to handle multiple parameterizations of the energy grid (e.g.,
 * arbitrary spacing needed for the Livermore sampling) and of the value
 * interpolation (e.g. log interpolation). It might also make sense to get rid
 * of the "prime energy" and just use log-log interpolation instead, or do a
 * piecewise change in the interpolation instead of storing the cross section
 * scaled by the energy.
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

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    const XsGridPointers& data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsGridCalculator.i.hh"
