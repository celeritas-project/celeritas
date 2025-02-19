//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.cc
//---------------------------------------------------------------------------//
#include "NonuniformGridBuilder.hh"

#include "corecel/grid/VectorUtils.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

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
    return this->insert_impl(grid, values, BC::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto NonuniformGridBuilder::operator()(SpanConstDbl grid, SpanConstDbl values)
    -> Grid
{
    return this->insert_impl(grid, values, BC::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto NonuniformGridBuilder::operator()(SpanConstFlt grid,
                                       SpanConstFlt values,
                                       BC bc) -> Grid
{
    CELER_EXPECT(bc != BC::size_);
    return this->insert_impl(grid, values, bc);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto NonuniformGridBuilder::operator()(SpanConstDbl grid,
                                       SpanConstDbl values,
                                       BC bc) -> Grid
{
    CELER_EXPECT(bc != BC::size_);
    return this->insert_impl(grid, values, bc);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from an imported physics vector.
 */
auto NonuniformGridBuilder::operator()(ImportPhysicsVector const& pvec) -> Grid
{
    CELER_EXPECT(pvec.vector_type == ImportPhysicsVectorType::free);
    return this->insert_impl(make_span(pvec.x), make_span(pvec.y), BC::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from an imported physics vector.
 */
auto NonuniformGridBuilder::operator()(ImportPhysicsVector const& pvec, BC bc)
    -> Grid
{
    CELER_EXPECT(bc != BC::size_);
    CELER_EXPECT(pvec.vector_type == ImportPhysicsVectorType::free);
    CELER_EXPECT(pvec.spline);
    return this->insert_impl(make_span(pvec.x), make_span(pvec.y), bc);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from container references.
 */
template<class T>
auto NonuniformGridBuilder::insert_impl(Span<T const> grid,
                                        Span<T const> values,
                                        BC bc) -> Grid
{
    CELER_EXPECT(grid.size() >= 2);
    CELER_EXPECT(grid.front() <= grid.back());
    CELER_EXPECT(values.size() == grid.size());

    Grid result;
    result.grid = reals_.insert_back(grid.begin(), grid.end());
    result.value = reals_.insert_back(values.begin(), values.end());
    if (bc != BC::size_)
    {
        // Calculate second derivatives for cubic spline interpolation
        CELER_ASSERT(is_monotonic_increasing(grid));
        auto deriv = SplineDerivCalculator(bc)(values_[result.grid],
                                               values_[result.value]);
        result.derivative = reals_.insert_back(deriv.begin(), deriv.end());
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
