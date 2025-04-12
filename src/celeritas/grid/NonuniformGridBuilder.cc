//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.cc
//---------------------------------------------------------------------------//
#include "NonuniformGridBuilder.hh"

#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to data that will be modified.
 */
NonuniformGridBuilder::NonuniformGridBuilder(Items<real_type>* reals)
    : values_(*reals), reals_{reals}
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
    return this->insert_impl(grid, values, {});
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto NonuniformGridBuilder::operator()(SpanConstDbl grid, SpanConstDbl values)
    -> Grid
{
    return this->insert_impl(grid, values, {});
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from an imported physics vector.
 */
auto NonuniformGridBuilder::operator()(inp::Grid const& grid) -> Grid
{
    return this->insert_impl(
        make_span(grid.x), make_span(grid.y), grid.interpolation);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from container references.
 */
template<class T>
auto NonuniformGridBuilder::insert_impl(Span<T const> grid,
                                        Span<T const> values,
                                        inp::Interpolation interp) -> Grid
{
    using BC = SplineDerivCalculator::BoundaryCondition;

    CELER_EXPECT(grid.size() >= 2);
    CELER_EXPECT(grid.front() <= grid.back());
    CELER_EXPECT(values.size() == grid.size());
    CELER_EXPECT(interp.type != InterpolationType::size_);

    if (interp.type == InterpolationType::poly_spline)
    {
        CELER_LOG(warning) << to_cstring(interp.type)
                           << " interpolation is not supported on a "
                              "nonuniform grid: defaulting to linear";
        interp.type = InterpolationType::linear;
    }
    else if (interp.type == InterpolationType::cubic_spline
             && values.size() < 5)
    {
        CELER_LOG(warning) << to_cstring(interp.type)
                           << " interpolation is not supported on a "
                              "grid with size "
                           << values.size() << ": defaulting to linear";
        interp.type = InterpolationType::linear;
    }

    Grid result;
    result.grid = reals_.insert_back(grid.begin(), grid.end());
    result.value = reals_.insert_back(values.begin(), values.end());
    if (interp.type == InterpolationType::cubic_spline)
    {
        // Calculate second derivatives for cubic spline interpolation
        CELER_ASSERT(interp.bc != BC::size_);
        CELER_ASSERT(is_monotonic_increasing(grid));
        auto deriv = SplineDerivCalculator(interp.bc)(values_[result.grid],
                                                      values_[result.value]);
        result.derivative = reals_.insert_back(deriv.begin(), deriv.end());
    }
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
