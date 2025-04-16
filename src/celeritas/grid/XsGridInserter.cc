//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/XsGridInserter.cc
//---------------------------------------------------------------------------//
#include "XsGridInserter.hh"

#include "corecel/Types.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
XsGridInserter::XsGridInserter(Values* reals, GridValues* grids)
    : values_(*reals), reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
auto XsGridInserter::operator()(inp::UniformGrid const& lower,
                                inp::UniformGrid const& upper) -> GridId
{
    CELER_EXPECT(lower || upper);

    XsGridRecord grid;
    if (lower)
    {
        grid.lower.grid = UniformGridData::from_bounds(lower.x, lower.y.size());
        grid.lower.value = reals_.insert_back(lower.y.begin(), lower.y.end());
        this->set_spline(lower, grid.lower);
    }
    if (upper)
    {
        grid.upper.grid = UniformGridData::from_bounds(upper.x, upper.y.size());
        grid.upper.value = reals_.insert_back(upper.y.begin(), upper.y.end());
        this->set_spline(upper, grid.upper);
    }
    CELER_ENSURE(grid);
    return grids_.push_back(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto XsGridInserter::operator()(inp::UniformGrid const& grid) -> GridId
{
    return (*this)(grid, {});
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives or set the polynomial order.
 */
void XsGridInserter::set_spline(inp::UniformGrid const& grid,
                                UniformGridRecord& data)
{
    CELER_EXPECT(grid);

    if (grid.interpolation.type == InterpolationType::cubic_spline)
    {
        if (data.value.size() < SplineDerivCalculator::min_grid_size())
        {
            CELER_LOG(warning)
                << to_cstring(grid.interpolation.type)
                << " interpolation is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return;
        }
        // Calculate second derivatives for cubic spline interpolation
        CELER_ASSERT(grid.interpolation.bc
                     != SplineDerivCalculator::BoundaryCondition::size_);
        auto deriv
            = SplineDerivCalculator(grid.interpolation.bc)(data, values_);
        data.derivative = reals_.insert_back(deriv.begin(), deriv.end());
    }
    else if (grid.interpolation.type == InterpolationType::poly_spline)
    {
        if (data.value.size() <= grid.interpolation.order)
        {
            CELER_LOG(warning)
                << to_cstring(grid.interpolation.type)
                << " interpolation with order " << grid.interpolation.order
                << " is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return;
        }
        CELER_ASSERT(grid.interpolation.order > 1);
        data.spline_order = grid.interpolation.order;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
