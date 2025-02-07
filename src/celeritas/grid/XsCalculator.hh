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

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate scaled cross sections on a uniform log grid.
 *
 * This cross section calculator uses the same representation and interpolation
 * as Geant4's physics tables for EM physics:
 * - The energy grid is uniformly spaced in log(E),
 * - Values greater than or equal to an index i' are scaled by E,
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
    XsCalculator calc_xs(xs_grid, xs_params.reals);
    real_type xs = calc_xs(particle.energy());
   \endcode
 */
class XsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = RealQuantity<XsGridData::EnergyUnits>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    XsCalculator(XsGridData const& grid, Values const& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    // Get the cross section at the given index
    inline CELER_FUNCTION real_type operator[](size_type index) const;

    // Get the minimum energy
    CELER_FUNCTION Energy energy_min() const
    {
        return Energy(std::exp(loge_grid_.front()));
    }

    // Get the maximum energy
    CELER_FUNCTION Energy energy_max() const
    {
        return Energy(std::exp(loge_grid_.back()));
    }

  private:
    XsGridData const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 */
CELER_FUNCTION
XsCalculator::XsCalculator(XsGridData const& grid, Values const& values)
    : data_(grid), reals_(values), loge_grid_(data_.log_energy)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 *
 * If needed, we can add a "log(energy/MeV)" accessor if we constantly reuse
 * that value and don't want to repeat the `std::log` operation.
 */
CELER_FUNCTION real_type XsCalculator::operator()(Energy energy) const
{
    real_type const loge = std::log(energy.value());

    auto calc_extrapolated = [this, &energy](size_type idx) {
        real_type result = this->get(idx);
        if (idx >= data_.prime_index)
        {
            result /= energy.value();
        }
        return result;
    };

    if (loge <= loge_grid_.front())
    {
        return calc_extrapolated(0);
    }
    if (loge >= loge_grid_.back())
    {
        return calc_extrapolated(loge_grid_.size() - 1);
    }

    // Locate the energy bin
    size_type lower_idx = loge_grid_.find(loge);
    CELER_ASSERT(lower_idx + 1 < loge_grid_.size());

    real_type lower_energy = std::exp(loge_grid_[lower_idx]);
    real_type upper_energy = std::exp(loge_grid_[lower_idx + 1]);

    real_type result;
    if (data_.derivative.empty())
    {
        // Interpolate *linearly* on energy
        real_type upper_xs = this->get(lower_idx + 1);
        if (lower_idx + 1 == data_.prime_index)
        {
            // Cross section data for the upper point is scaled by E: calculate
            // the unscaled value
            upper_xs /= upper_energy;
        }

        result = LinearInterpolator<real_type>(
            {lower_energy, this->get(lower_idx)},
            {upper_energy, upper_xs})(value_as<Energy>(energy));

        if (lower_idx >= data_.prime_index)
        {
            result /= energy.value();
        }
    }
    else
    {
        // Use cubic spline interpolation
        CELER_ASSERT(data_.prime_index == XsGridData::no_scaling());
        CELER_ASSERT(lower_idx + 1 < data_.derivative.size());
        real_type lower_deriv = reals_[data_.derivative[lower_idx]];
        real_type upper_deriv = reals_[data_.derivative[lower_idx + 1]];

        result = SplineInterpolator<real_type>(
            {lower_energy, (*this)[lower_idx], lower_deriv},
            {upper_energy, (*this)[lower_idx + 1], upper_deriv})(
            value_as<Energy>(energy));
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the cross section at the given index.
 */
CELER_FUNCTION real_type XsCalculator::operator[](size_type index) const
{
    real_type result = this->get(index);
    if (index >= data_.prime_index)
    {
        result /= std::exp(loge_grid_[index]);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the raw cross section data at a particular index.
 */
CELER_FUNCTION real_type XsCalculator::get(size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
