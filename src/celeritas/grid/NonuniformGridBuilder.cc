//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.cc
//---------------------------------------------------------------------------//
#include "NonuniformGridBuilder.hh"

#include "corecel/grid/VectorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to data that will be modified.
 */
NonuniformGridBuilder::NonuniformGridBuilder(Items<real_type>* reals)
    : NonuniformGridBuilder(reals, BC::size_)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to data and boundary condtions.
 */
NonuniformGridBuilder::NonuniformGridBuilder(Items<real_type>* reals, BC bc)
    : values_(*reals), reals_{reals}, bc_(bc)
{
    CELER_EXPECT(reals);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto NonuniformGridBuilder::operator()(SpanConstFlt grid, SpanConstFlt values)
    -> Grid
{
    return this->insert_impl(grid, values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto NonuniformGridBuilder::operator()(SpanConstDbl grid, SpanConstDbl values)
    -> Grid
{
    return this->insert_impl(grid, values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from an imported physics vector.
 */
auto NonuniformGridBuilder::operator()(inp::Grid const& grid) -> Grid
{
    return this->insert_impl(make_span(grid.x), make_span(grid.y));
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from container references.
 */
template<class T>
auto NonuniformGridBuilder::insert_impl(Span<T const> grid,
                                        Span<T const> values) -> Grid
{
    CELER_EXPECT(grid.size() >= 2);
    CELER_EXPECT(grid.front() <= grid.back());
    CELER_EXPECT(values.size() == grid.size());

    Grid result;
    result.grid = reals_.insert_back(grid.begin(), grid.end());
    result.value = reals_.insert_back(values.begin(), values.end());
    if (bc_ != BC::size_)
    {
        // Calculate second derivatives for cubic spline interpolation
        CELER_ASSERT(is_monotonic_increasing(grid));
        auto deriv = SplineDerivCalculator(bc_)(values_[result.grid],
                                                values_[result.value]);
        result.derivative = reals_.insert_back(deriv.begin(), deriv.end());
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
