//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.cc
//---------------------------------------------------------------------------//
#include "RangeGridCalculator.hh"

#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with energy loss table.
 */
RangeGridCalculator::RangeGridCalculator(XsGridData const& grid,
                                         Values const& reals,
                                         size_type spline_order)
    : calc_dedx_(grid, reals, spline_order), loge_grid_(grid.log_energy)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the range from the energy loss.
 *
 * This assumes the same log energy grid is used for range and energy loss.
 */
std::vector<real_type> RangeGridCalculator::operator()() const
{
    using Energy = units::MevEnergy;

    constexpr real_type delta = 1 / real_type(integration_substeps());

    std::vector<real_type> result(loge_grid_.size());

    CELER_ASSERT(calc_dedx_[0] > 0);
    real_type integral = 2 * std::exp(loge_grid_[0]) / calc_dedx_[0];
    result[0] = integral;
    for (size_type i = 1; i < loge_grid_.size(); ++i)
    {
        real_type energy_lower = std::exp(loge_grid_[i - 1]);
        real_type energy_upper = std::exp(loge_grid_[i]);
        real_type delta_energy = (energy_upper - energy_lower) * delta;
        real_type energy = energy_upper + 0.5 * delta_energy;
        for (size_type j = 0; j < integration_substeps(); ++j)
        {
            energy -= delta_energy;
            real_type dedx = calc_dedx_(Energy(energy));
            CELER_ASSERT(dedx > 0);
            integral += delta_energy / dedx;
        }
        result[i] = integral;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
