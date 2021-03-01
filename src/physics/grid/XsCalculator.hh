//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Quantity.hh"
#include "XsGridInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross sections on a uniform log grid.
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
    XsCalculator calc_xs(xs_grid, xs_params.reals);
    real_type xs = calc_xs(particle);
   \endcode
 */
class XsCalculator
{
  public:
    //!@{
    //! Type aliases
    using Energy = Quantity<XsGridData::EnergyUnits>;
    using Values = Pie<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    XsCalculator(const XsGridData& grid, const Values& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    const XsGridData& data_;
    const Values&     reals_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "XsCalculator.i.hh"
