//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CalculatorTestBase.cc
//---------------------------------------------------------------------------//
#include "CalculatorTestBase.hh"

#include <cmath>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct from grid bounds and cross section values.
 */
void CalculatorTestBase::build(GridInput grid, GridInput grid_scaled)
{
    return this->build_impl(std::move(grid), std::move(grid_scaled), BC::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from grid bounds and cross section values.
 */
void CalculatorTestBase::build_spline(GridInput grid,
                                      GridInput grid_scaled,
                                      BC bc)
{
    CELER_EXPECT(bc != BC::size_);
    return this->build_impl(std::move(grid), std::move(grid_scaled), bc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct without scaled values.
 */
void CalculatorTestBase::build(GridInput grid)
{
    return this->build_impl(std::move(grid), BC::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct without scaled values.
 */
void CalculatorTestBase::build_spline(GridInput grid, BC bc)
{
    CELER_EXPECT(bc != BC::size_);
    return this->build_impl(std::move(grid), bc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from grid bounds and cross section values.
 */
void CalculatorTestBase::build_impl(GridInput grid, GridInput grid_scaled, BC bc)
{
    CELER_EXPECT((!grid.value.empty() || !grid_scaled.value.empty())
                 && (grid.value.empty() || grid_scaled.value.empty()
                     || grid.emax == grid_scaled.emin));
    CELER_EXPECT(grid.value.empty()
                 || (grid.value.size() >= 2 && grid.emin > 0
                     && grid.emax > grid.emin && grid.spline_order > 0));
    CELER_EXPECT(grid_scaled.value.empty()
                 || (grid_scaled.value.size() >= 2 && grid_scaled.emin > 0
                     && grid_scaled.emax > grid_scaled.emin
                     && grid_scaled.spline_order > 0));

    value_storage_ = {};

    if (!grid.value.empty())
    {
        this->build_grid(data_.lower, grid, bc);
    }
    if (!grid_scaled.value.empty())
    {
        // Scale cross section values by energy
        auto loge_grid
            = UniformGridData::from_bounds(std::log(grid_scaled.emin),
                                           std::log(grid_scaled.emax),
                                           grid_scaled.value.size());
        UniformGrid loge{loge_grid};
        for (auto i : range(loge.size()))
        {
            grid_scaled.value[i] *= std::exp(loge[i]);
        }
        this->build_grid(data_.upper, grid_scaled, bc);
    }

    value_ref_ = value_storage_;

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct without scaled values.
 */
void CalculatorTestBase::build_impl(GridInput grid, BC bc)
{
    this->build_impl(grid, {}, bc);
}

//---------------------------------------------------------------------------//
/*!
 * Build a uniform grid.
 */
void CalculatorTestBase::build_grid(UniformGridRecord& data,
                                    GridInput const& grid,
                                    BC bc)
{
    CollectionBuilder build(&value_storage_);

    data.grid = UniformGridData::from_bounds(
        std::log(grid.emin), std::log(grid.emax), grid.value.size());
    data.value = build.insert_back(grid.value.begin(), grid.value.end());
    data.spline_order = grid.spline_order;

    if (bc != BC::size_)
    {
        Data value_ref{value_storage_};
        auto deriv = SplineDerivCalculator(bc)(data, value_ref);
        data.derivative = build.insert_back(deriv.begin(), deriv.end());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
