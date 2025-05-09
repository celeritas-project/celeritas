//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.cc
//---------------------------------------------------------------------------//
#include "RangeGridCalculator.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"

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
inp::UniformGrid
RangeGridCalculator::operator()(inp::UniformGrid const& dedx_grid) const
{
    using HostValues = Collection<real_type, Ownership::value, MemSpace::host>;
    using HostCRef
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;

    HostValues host_values;
    HostCRef host_ref;

    UniformGridRecord data;
    data.grid = UniformGridData::from_bounds(dedx_grid.x, dedx_grid.y.size());

    auto calc_dedx = [&] {
        // Create a copy of the grid data, with the derivatives if needed
        CollectionBuilder build(&host_values);
        data.value = build.insert_back(dedx_grid.y.begin(), dedx_grid.y.end());
        host_ref = host_values;

        if (dedx_grid.y.size() >= 5 && bc_ != BC::size_)
        {
            // Calculate the second derivatives for cubic spline interpolation
            auto deriv = SplineDerivCalculator(bc_)(data, host_ref);
            data.derivative = build.insert_back(deriv.begin(), deriv.end());
            host_ref = host_values;
        }
        return UniformLogGridCalculator(data, host_ref);
    }();

    UniformGrid loge_grid(data.grid);

    inp::UniformGrid result;
    result.x = dedx_grid.x;
    result.y.resize(dedx_grid.y.size());
    result.interpolation = dedx_grid.interpolation;
    if (result.interpolation.type == InterpolationType::poly_spline)
    {
        CELER_LOG(warning) << InterpolationType::poly_spline
                           << " interpolation is not supported for range "
                              "or inverse range: defaulting to linear";
        result.interpolation.type = InterpolationType::linear;
    }

    constexpr real_type delta
        = 1 / static_cast<real_type>(integration_substeps());

    CELER_ASSERT(calc_dedx[0] > 0);
    real_type cum_range = 2 * std::exp(loge_grid[0]) / calc_dedx[0];
    result.y[0] = cum_range;

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
        result.y[i] = cum_range;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
