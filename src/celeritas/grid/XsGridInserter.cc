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
auto XsGridInserter::operator()(UniformGridData const& lower_grid,
                                SpanConstDbl lower_values,
                                inp::Interpolation lower_interp,
                                UniformGridData const& upper_grid,
                                SpanConstDbl upper_values,
                                inp::Interpolation upper_interp) -> GridId
{
    return this->insert(lower_grid,
                        lower_values,
                        lower_interp,
                        upper_grid,
                        upper_values,
                        upper_interp);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
auto XsGridInserter::operator()(UniformGridData const& lower_grid,
                                SpanConstFlt lower_values,
                                inp::Interpolation lower_interp,
                                UniformGridData const& upper_grid,
                                SpanConstFlt upper_values,
                                inp::Interpolation upper_interp) -> GridId
{
    return this->insert(lower_grid,
                        lower_values,
                        lower_interp,
                        upper_grid,
                        upper_values,
                        upper_interp);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto XsGridInserter::operator()(UniformGridData const& grid,
                                SpanConstDbl values,
                                inp::Interpolation interp) -> GridId
{
    return (*this)(grid, values, interp, UniformGridData{}, {}, interp);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto XsGridInserter::operator()(UniformGridData const& grid,
                                SpanConstFlt values,
                                inp::Interpolation interp) -> GridId
{
    return (*this)(grid, values, interp, UniformGridData{}, {}, interp);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
template<class T>
auto XsGridInserter::insert(UniformGridData const& lower_grid,
                            Span<T const> lower_values,
                            inp::Interpolation lower_interp,
                            UniformGridData const& upper_grid,
                            Span<T const> upper_values,
                            inp::Interpolation upper_interp) -> GridId
{
    CELER_EXPECT(lower_grid || upper_grid);
    CELER_EXPECT(!lower_grid || lower_grid.size == lower_values.size());
    CELER_EXPECT(!upper_grid || upper_grid.size == upper_values.size());

    XsGridRecord grid;
    grid.lower.grid = lower_grid;
    grid.upper.grid = upper_grid;
    grid.lower.value
        = reals_.insert_back(lower_values.begin(), lower_values.end());
    grid.upper.value
        = reals_.insert_back(upper_values.begin(), upper_values.end());

    set_spline(lower_grid, lower_interp, grid.lower);
    set_spline(upper_grid, upper_interp, grid.upper);

    CELER_ENSURE(grid);
    return grids_.push_back(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives or set the polynomial order.
 */
void XsGridInserter::set_spline(UniformGridData const& grid,
                                inp::Interpolation const& interp,
                                UniformGridRecord& data)
{
    if (!grid)
    {
        return;
    }
    if (interp.type == InterpolationType::cubic_spline)
    {
        if (data.value.size() < 5)
        {
            CELER_LOG(warning)
                << to_cstring(interp.type)
                << " interpolation is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return;
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
            return;
        }
        CELER_ASSERT(interp.order > 1);
        data.spline_order = interp.order;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
