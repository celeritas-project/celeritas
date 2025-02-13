//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.cc
//---------------------------------------------------------------------------//
#include "RangeGridCalculator.hh"

#include "corecel/data/CollectionBuilder.hh"

#include "UniformLogGridCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Default constructor.
 */
RangeGridCalculator::RangeGridCalculator() : RangeGridCalculator(BC::size_) {}

//---------------------------------------------------------------------------//
/*!
 * Construct with boundary conditions for spline interpolation.
 */
RangeGridCalculator::RangeGridCalculator(BC bc) : bc_(bc) {}

//---------------------------------------------------------------------------//
/*!
 * Calculate the range from the energy loss.
 *
 * This assumes the same log energy grid is used for range and energy loss.
 */
auto RangeGridCalculator::operator()(UniformGridRecord const& orig_data,
                                     Values const& orig_reals) const -> VecReal
{
    using HostValues = Collection<real_type, Ownership::value, MemSpace::host>;

    HostValues host_reals;
    Values reals;
    UniformGridRecord data;

    auto calc_dedx = [&] {
        if (orig_data.value.size() < 5 || bc_ == BC::size_)
        {
            // Use linear interpolation
            data = orig_data;
            data.derivative = {};
            return UniformLogGridCalculator(data, orig_reals);
        }
        else if (orig_data.derivative.empty())
        {
            // Calculate the second derivatives for cubic spline interpolation
            auto deriv = SplineDerivCalculator(bc_)(orig_data, orig_reals);

            // Create a copy of the grid data with the derivatives
            CollectionBuilder build(&host_reals);
            data.grid = orig_data.grid;
            data.value = build.insert_back(orig_reals[orig_data.value].begin(),
                                           orig_reals[orig_data.value].end());
            data.derivative = build.insert_back(deriv.begin(), deriv.end());
            reals = host_reals;
            return UniformLogGridCalculator(data, reals);
        }
        // The derivatives have already been calculated
        return UniformLogGridCalculator(orig_data, orig_reals);
    }();

    UniformGrid loge_grid(orig_data.grid);
    VecReal result(loge_grid.size());

    constexpr real_type delta
        = 1 / static_cast<real_type>(integration_substeps());

    CELER_ASSERT(calc_dedx[0] > 0);
    real_type cum_range = 2 * std::exp(loge_grid[0]) / calc_dedx[0];
    result[0] = cum_range;

    // Integrate the range from the energy loss
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

            // Spline interpolation can exhibit oscillations that greatly
            // affect the accuracy when the number of grid points is small and
            // the scale of the x grid is large
            CELER_VALIDATE(dedx > 0,
                           << "negative value in range calculation: the "
                              "interpolation method may be unstable");
            cum_range += delta_energy / dedx;
        }
        result[i] = cum_range;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
