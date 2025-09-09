//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.cc
//---------------------------------------------------------------------------//
#include "NonuniformGridBuilder.hh"

#include "corecel/Types.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/io/EnumStringMapper.hh"

#include "detail/GridUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
NonuniformGridBuilder::NonuniformGridBuilder(Values* reals)
    : values_(reals), reals_(reals)
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

    CELER_VALIDATE(grid.interpolation.type != InterpolationType::poly_spline,
                   << grid.interpolation.type
                   << " interpolation is not supported on a nonuniform grid");

    NonuniformGridRecord data;
    data.grid = reals_.insert_back(grid.x.begin(), grid.x.end());
    data.value = reals_.insert_back(grid.y.begin(), grid.y.end());
    detail::set_spline(values_, reals_, grid.interpolation, data);

    CELER_ENSURE(data);
    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
