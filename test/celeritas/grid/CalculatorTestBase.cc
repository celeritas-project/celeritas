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
void CalculatorTestBase::build(inp::UniformGrid lower, inp::UniformGrid upper)
{
    return this->build_impl(lower, upper, false);
}

//---------------------------------------------------------------------------//
/*!
 * Construct without scaled values.
 */
void CalculatorTestBase::build(inp::UniformGrid grid)
{
    return this->build_impl(grid, {}, false);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an inverted grid.
 */
void CalculatorTestBase::build_inverted(inp::UniformGrid grid)
{
    return this->build_impl(grid, {}, true);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from grid bounds and cross section values.
 */
void CalculatorTestBase::build_impl(inp::UniformGrid lower,
                                    inp::UniformGrid upper,
                                    bool invert)
{
    CELER_EXPECT(
        (lower || upper)
        && (!lower || !upper || lower.x[Bound::hi] == upper.x[Bound::lo]));
    CELER_EXPECT(!lower || (lower.y.size() >= 2 && lower.x[Bound::lo] > 0));
    CELER_EXPECT(!upper || (upper.y.size() >= 2 && upper.x[Bound::lo] > 0));

    value_storage_ = {};

    if (lower)
    {
        this->build_grid(data_.lower, lower, invert);
    }
    if (upper)
    {
        // Scale cross section values by energy
        auto loge_grid = UniformGridData::from_bounds(
            {std::log(upper.x[Bound::lo]), std::log(upper.x[Bound::hi])},
            upper.y.size());
        UniformGrid loge{loge_grid};
        for (auto i : range(loge.size()))
        {
            upper.y[i] *= std::exp(loge[i]);
        }
        this->build_grid(data_.upper, upper, invert);
    }

    value_ref_ = value_storage_;

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build a uniform grid.
 */
void CalculatorTestBase::build_grid(UniformGridRecord& data,
                                    inp::UniformGrid const& grid,
                                    bool invert)
{
    CollectionBuilder build(&value_storage_);

    data.grid = UniformGridData::from_bounds(
        {std::log(grid.x[Bound::lo]), std::log(grid.x[Bound::hi])},
        grid.y.size());
    data.value = build.insert_back(grid.y.begin(), grid.y.end());
    data.spline_order = grid.interpolation.order;

    if (grid.interpolation.type == InterpolationType::cubic_spline)
    {
        Data value_ref{value_storage_};
        SplineDerivCalculator calc(grid.interpolation.bc);
        auto deriv = invert ? calc.calc_from_inverse(data, value_ref)
                            : calc(data, value_ref);
        data.derivative = build.insert_back(deriv.begin(), deriv.end());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
