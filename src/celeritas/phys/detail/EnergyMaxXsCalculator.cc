//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/EnergyMaxXsCalculator.cc
//---------------------------------------------------------------------------//
#include "EnergyMaxXsCalculator.hh"

#include <algorithm>
#include <cmath>

#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options and process.
 */
EnergyMaxXsCalculator::EnergyMaxXsCalculator(PhysicsOptions const& opts,
                                             Process const& proc)
    : use_integral_xs_(!opts.disable_integral_xs && proc.supports_integral_xs())
    , is_annihilation_(dynamic_cast<EPlusAnnihilationProcess const*>(&proc))
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy of the maximum cross section.
 *
 * The annihilation process calculates cross sections on the fly so does not
 * have a macroscopic cross section grid: its cross section is maximum at zero
 * and decreases with increasing energy.
 */
real_type EnergyMaxXsCalculator::operator()(inp::XsGrid const& macro_xs) const
{
    CELER_EXPECT(use_integral_xs_);
    CELER_EXPECT(macro_xs || is_annihilation_);

    real_type result = 0;
    real_type max_xs = 0;
    if (auto const& grid = macro_xs.lower)
    {
        // Find the index of the maximum cross section
        size_type max_el = std::distance(
            grid.y.begin(), std::max_element(grid.y.begin(), grid.y.end()));

        // Get the energy and cross section value at that index
        auto loge_data = UniformGridData::from_bounds(grid.x, grid.y.size());
        result = std::exp(UniformGrid(loge_data)[max_el]);
        max_xs = grid.y[max_el];
    }
    if (auto const& grid = macro_xs.upper)
    {
        // Find the index of the maximum cross section, scaling the cross
        // section by the energy
        auto loge_data = UniformGridData::from_bounds(grid.x, grid.y.size());
        UniformGrid loge_grid(loge_data);
        for (auto i : range(grid.y.size()))
        {
            real_type energy = std::exp(loge_grid[i]);
            auto xs = grid.y[i] / energy;
            if (xs > max_xs)
            {
                max_xs = xs;
                result = energy;
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
