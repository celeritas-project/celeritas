//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformLogGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/SplineInterpolator.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate values on a uniform log energy grid.
 *
 * Note that linear interpolation is applied with energy points, not log-energy
 * points.
 *
 * \code
    UniformLogGridCalculator calc(grid, params.reals);
    real_type y = calc(particle.energy());
   \endcode
 */
class UniformLogGridCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    UniformLogGridCalculator(UniformGridRecord const& grid,
                             Values const& reals);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    // Get the tabulated value at the given index
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
    UniformGridRecord const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from uniform grid data.
 */
CELER_FUNCTION
UniformLogGridCalculator::UniformLogGridCalculator(UniformGridRecord const& grid,
                                                   Values const& reals)
    : data_(grid), reals_(reals), loge_grid_(data_.grid)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate the value at the given energy.
 */
CELER_FUNCTION real_type UniformLogGridCalculator::operator()(Energy energy) const
{
    real_type const loge = std::log(value_as<Energy>(energy));

    // Snap out-of-bounds values to closest grid points
    if (loge <= loge_grid_.front())
    {
        return (*this)[0];
    }
    if (loge >= loge_grid_.back())
    {
        return (*this)[loge_grid_.size() - 1];
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
        result = LinearInterpolator<real_type>(
            {lower_energy, (*this)[lower_idx]},
            {upper_energy, (*this)[lower_idx + 1]})(value_as<Energy>(energy));
    }
    else
    {
        // Use cubic spline interpolation
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
 * Get the tabulated value at the given index.
 */
CELER_FUNCTION real_type UniformLogGridCalculator::operator[](size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
