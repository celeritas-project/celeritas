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
auto UniformGridInserter::operator()(UniformGridData const& grid,
                                     SpanConstDbl values,
                                     inp::Interpolation interp) -> GridId
{
    return this->insert(grid, values, interp);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
auto UniformGridInserter::operator()(UniformGridData const& grid,
                                     SpanConstFlt values,
                                     inp::Interpolation interp) -> GridId
{
    return this->insert(grid, values, interp);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
template<class T>
auto UniformGridInserter::insert(UniformGridData const& grid,
                                 Span<T const> values,
                                 inp::Interpolation interp) -> GridId
{
    CELER_EXPECT(grid);
    CELER_EXPECT(grid.size == values.size());

    UniformGridRecord data;
    data.grid = grid;
    data.value = reals_.insert_back(values.begin(), values.end());
    if (interp.type == InterpolationType::cubic_spline)
    {
        if (data.value.size() < 5)
        {
            CELER_LOG(warning)
                << to_cstring(interp.type)
                << " interpolation is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return grids_.push_back(data);
        }

        // Calculate second derivatives for cubic spline interpolation
        CELER_ASSERT(interp.bc
                     != SplineDerivCalculator::BoundaryCondition::size_);
        ValuesRef ref(values_);
        auto deriv = SplineDerivCalculator(interp.bc)(data, ref);
        data.derivative = reals_.insert_back(deriv.begin(), deriv.end());
    }
    else if (interp.type == InterpolationType::poly_spline)
    {
        if (interp.order >= data.value.size())
        {
            CELER_LOG(warning)
                << to_cstring(interp.type) << " interpolation with order "
                << interp.order << " is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return grids_.push_back(data);
        }
        CELER_ASSERT(interp.order > 1);
        data.spline_order = interp.order;
    }
    return grids_.push_back(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
