//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGridInserter.cc
//---------------------------------------------------------------------------//
#include "UniformGridInserter.hh"

#include "corecel/Types.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
UniformGridInserter::UniformGridInserter(Values* reals, GridValues* grids)
    : values_(*reals), reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
auto UniformGridInserter::operator()(inp::UniformGrid const& grid) -> GridId
{
    CELER_EXPECT(grid);

    UniformGridRecord data;
    data.grid = UniformGridData::from_bounds(grid.x, grid.y.size());
    data.value = reals_.insert_back(grid.y.begin(), grid.y.end());
    if (grid.interpolation.type == InterpolationType::cubic_spline)
    {
        if (data.value.size() < SplineDerivCalculator::min_grid_size())
        {
            CELER_LOG(warning)
                << to_cstring(grid.interpolation.type)
                << " interpolation is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return grids_.push_back(data);
        }
        using ValuesRef
            = Collection<real_type, Ownership::const_reference, MemSpace::host>;

        // Calculate second derivatives for cubic spline interpolation
        CELER_ASSERT(grid.interpolation.bc
                     != SplineDerivCalculator::BoundaryCondition::size_);
        ValuesRef values(values_);
        auto deriv = SplineDerivCalculator(grid.interpolation.bc)(data, values);
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
            return grids_.push_back(data);
        }
        CELER_ASSERT(grid.interpolation.order > 1);
        data.spline_order = grid.interpolation.order;
    }
    return grids_.push_back(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
