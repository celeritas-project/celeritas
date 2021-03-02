//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InverseRangeCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Quantity.hh"
#include "NonuniformGrid.hh"
#include "XsGridInterface.hh"
#include "UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the energy that would limit a particle to a particular range.
 *
 * This should provide the inverse of the result of \c RangeCalculator. The
 * given \c range is not allowed to be greater than the maximum range in the
 * physics data.
 *
 * The range must be monotonically increasing in energy, since it's defined as
 * the integral of the inverse of the stopping power (which is always
 * positive). For ranges shorter than the minimum energy in the table, the
 * resulting energy is scaled:
 * \f[
    E = E_\textrm{min}} \left( \frac{r}{r_\textrm{min}} \right)^2
 * \f]
 * This scaling is the inverse of the off-the-end energy scaling in the
 * RangeCalculator.
 */
class InverseRangeCalculator
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
    InverseRangeCalculator(const XsGridData& grid, const Values& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION Energy operator()(real_type range) const;

  private:
    UniformGrid               log_energy_;
    NonuniformGrid<real_type> range_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "InverseRangeCalculator.i.hh"
