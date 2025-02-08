//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.cc
//---------------------------------------------------------------------------//
#include "RangeGridCalculator.hh"

#include "corecel/data/CollectionBuilder.hh"

#include "EnergyLossCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with boundary conditions for spline interpolation.
 */
RangeGridCalculator::RangeGridCalculator(BC bc) : bc_(bc)
{
    CELER_EXPECT(bc_ != BC::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the range from the energy loss.
 *
 * This assumes the same log energy grid is used for range and energy loss.
 */
auto RangeGridCalculator::operator()(XsGridData const& orig_data,
                                     Values const& orig_reals) const -> VecReal
{
    using HostValues = Collection<real_type, Ownership::value, MemSpace::host>;

    CELER_EXPECT(orig_data.prime_index == XsGridData::no_scaling());

    XsGridData data;
    HostValues host_reals;

    auto calc_dedx = [&] {
        if (!orig_data.derivative.empty() || orig_data.value.size() < 5)
        {
            return EnergyLossCalculator(orig_data, orig_reals);
        }

        // Calculate the second derivatives for cubic spline interpolation
        auto deriv = SplineDerivCalculator(bc_)(orig_data, orig_reals);

        // Create a copy of the grid data with the derivatives
        CollectionBuilder build(&host_reals);
        data.log_energy = orig_data.log_energy;
        data.value = build.insert_back(orig_reals[orig_data.value].begin(),
                                       orig_reals[orig_data.value].end());
        data.derivative = build.insert_back(deriv.begin(), deriv.end());
        Values reals(host_reals);
        return EnergyLossCalculator(data, reals);
    }();

    UniformGrid loge_grid(orig_data.log_energy);
    VecReal result(loge_grid.size());

    constexpr real_type delta = 1 / real_type(integration_substeps());

    CELER_ASSERT(calc_dedx[0] > 0);
    real_type cum_range = 2 * std::exp(loge_grid[0]) / calc_dedx[0];
    result[0] = cum_range;

    for (size_type i = 1; i < loge_grid.size(); ++i)
    {
        real_type energy_lower = std::exp(loge_grid[i - 1]);
        real_type energy_upper = std::exp(loge_grid[i]);
        real_type delta_energy = (energy_upper - energy_lower) * delta;
        real_type energy = energy_upper + 0.5 * delta_energy;
        for (size_type j = 0; j < integration_substeps(); ++j)
        {
            energy -= delta_energy;
            real_type dedx = calc_dedx(units::MevEnergy(energy));
            CELER_ASSERT(dedx > 0);
            cum_range += delta_energy / dedx;
        }
        result[i] = cum_range;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
