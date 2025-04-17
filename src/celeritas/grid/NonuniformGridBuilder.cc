//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.cc
//---------------------------------------------------------------------------//
#include "NonuniformGridBuilder.hh"

#include "corecel/Types.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
NonuniformGridBuilder::NonuniformGridBuilder(Values* reals)
    : values_(*reals), reals_(reals)
{
    CELER_EXPECT(reals);
}

//---------------------------------------------------------------------------//
/*!
 * Add a nonuniform grid.
 */
auto NonuniformGridBuilder::operator()(inp::Grid const& grid) -> Grid
{
    CELER_EXPECT(grid);

    NonuniformGridRecord data;
    data.grid = reals_.insert_back(grid.x.begin(), grid.x.end());
    data.value = reals_.insert_back(grid.y.begin(), grid.y.end());
    if (grid.interpolation.type == InterpolationType::cubic_spline)
    {
        if (data.value.size() < SplineDerivCalculator::min_grid_size())
        {
            CELER_LOG(warning)
                << to_cstring(grid.interpolation.type)
                << " interpolation is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return data;
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

    CELER_ENSURE(data);
    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
