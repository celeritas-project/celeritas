//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/DerivativeGridBuilder.cc
//---------------------------------------------------------------------------//
#include "DerivativeGridBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 *
 * The provided epsilon value determines the size of the epsilon-neighborhood
 * around grid-points where the derivative might not be well-defined.
 */
DerivativeGridBuilder::DerivativeGridBuilder(Values* reals, real_type epsilon)
    : epsilon_(epsilon), values_(reals), reals_(reals)
{
    CELER_EXPECT(reals);
    CELER_EXPECT(epsilon > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the derivative grid of an imported grid.
 *
 * For each grid-point x in the input, the (x-epsilon, x+epsilon) is
 * constructed. Outside the neighborhood, the derivative is well defined,
 * whereas in the neighborhood it is interpolated between the endpoints of the
 * neighborhood.
 *
 * Since only linearly interpolated \c NonuniformGridRecord are supported, the
 * derivative outside the epsilon-neighborhood is just the interpolated slope
 * between the grid-points.
 */
auto DerivativeGridBuilder::operator()(inp::Grid const&) -> Grid
{
    CELER_NOT_IMPLEMENTED("temp");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
