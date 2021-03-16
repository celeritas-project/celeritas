//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RangeCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Quantity.hh"
#include "XsGridInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate range on a uniform log grid.
 *
 * \code
    RangeCalculator calc_range(xs_grid, xs_params.reals);
    real_type range = calc_range(particle);
   \endcode
 *
 * Below the minimum tabulated energy, the range is scaled:
 * \f[
    r = r_\textrm{min} \sqrt{\frac{E}{E_\textrm{min}}}
 * \f]
 */
class RangeCalculator
{
  public:
    //!@{
    //! Type aliases
    using Energy = Quantity<XsGridData::EnergyUnits>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    RangeCalculator(const XsGridData& grid, const Values& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    const XsGridData& data_;
    const Values&     reals_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RangeCalculator.i.hh"
