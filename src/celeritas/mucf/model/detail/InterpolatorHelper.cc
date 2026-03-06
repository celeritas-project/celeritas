//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/InterpolatorHelper.cc
//---------------------------------------------------------------------------//
#include "InterpolatorHelper.hh"

#include "corecel/Assert.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with grid input data.
 */
InterpolatorHelper::InterpolatorHelper(inp::Grid const& input)
    : grid_record_(NonuniformGridBuilder(&reals_)(input))
{
    CELER_EXPECT(input);
    CELER_ENSURE(grid_record_);
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate data at given point.
 */
real_type InterpolatorHelper::operator()(real_type value) const
{
    NonuniformGridCalculator interpolate{grid_record_, make_ref(reals_)};
    return interpolate(value);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
