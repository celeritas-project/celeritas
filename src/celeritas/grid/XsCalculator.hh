//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/XsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/SplineInterpolator.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/Quantity.hh"

#include "SplineCalculator.hh"
#include "UniformLogGridCalculator.hh"
#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate scaled cross sections.
 *
 * This cross section calculator uses the same representation and interpolation
 * as Geant4's physics tables for EM physics:
 * - The energy grid is uniformly spaced in log(E),
 * - Values greater than or equal to an index i' are scaled by E and are stored
 *   on a separate energy grid also uniformly spaced in log(E) but not
 *   necessarily with the same spacing,
 * - Linear interpolation between energy points is used to calculate the final
 *   value, and
 * - If the energy is at or above the i' index, the final result is scaled by
 *   1/E.
 *
 * This scaling and interpolation exactly reproduces functions
 * \f$ f(E) \sim a E + b \f$ below the E' threshold and
 * \f$ f(E) \sim \frac{a'}{E} + b' \f$ above that threshold.
 *
 * Note that linear interpolation is applied with energy points, not log-energy
 * points.
 *
 * \code
    XsCalculator calc_xs(grid, params.reals);
    real_type xs = calc_xs(particle.energy());
   \endcode
 */
class XsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = RealQuantity<XsGridRecord::EnergyUnits>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    XsCalculator(XsGridRecord const& grid, Values const& reals);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    // Get the minimum energy
    inline CELER_FUNCTION Energy energy_min() const;

    // Get the maximum energy
    inline CELER_FUNCTION Energy energy_max() const;

  private:
    XsGridRecord const& data_;
    Values const& reals_;

    inline CELER_FUNCTION bool use_scaled(Energy energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 */
CELER_FUNCTION
XsCalculator::XsCalculator(XsGridRecord const& grid, Values const& reals)
    : data_(grid), reals_(reals)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section using linear or spline interpolation.
 */
CELER_FUNCTION real_type XsCalculator::operator()(Energy energy) const
{
    bool use_scaled = this->use_scaled(energy);
    auto const& grid = use_scaled ? data_.upper : data_.lower;

    real_type result;
    if (grid.spline_order == 1)
    {
        // Linear or cubic spline interpolation
        result = UniformLogGridCalculator(grid, reals_)(energy);
    }
    else
    {
        // Spline interpolation without continuous derivatives
        result = SplineCalculator(grid, reals_)(energy);
    }
    if (use_scaled)
    {
        result /= value_as<Energy>(energy);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the minimum energy.
 */
CELER_FUNCTION auto XsCalculator::energy_min() const -> Energy
{
    return Energy(std::exp(data_.lower ? data_.lower.grid.front
                                       : data_.upper.grid.front));
}

//---------------------------------------------------------------------------//
/*!
 * Get the maximum energy.
 */
CELER_FUNCTION auto XsCalculator::energy_max() const -> Energy
{
    return Energy(
        std::exp(data_.upper ? data_.upper.grid.back : data_.lower.grid.back));
}

//---------------------------------------------------------------------------//
/*!
 * Whether to use the scaled cross section grid.
 */
CELER_FUNCTION bool XsCalculator::use_scaled(Energy energy) const
{
    return !data_.lower
           || (data_.upper
               && std::log(value_as<Energy>(energy)) >= data_.upper.grid.front);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
