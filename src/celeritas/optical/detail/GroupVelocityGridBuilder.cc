//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/GroupVelocityGridBuilder.cc
//---------------------------------------------------------------------------//
#include "GroupVelocityGridBuilder.hh"

#include "corecel/Assert.hh"
#include "corecel/grid/DerivativeGridCalculator.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build a group-velocity grid from a refractive-index grid.
 *
 * The input grid supplies the samples used to calculate \f$dn/dE\f$. The
 * calculator interpolates \f$n(E)\f$ at the derivative-grid energies.
 *
 * The group velocity iss
 * \f[
 * v_g = \frac{c}{n(E) + E\frac{dn}{dE}}.
 * \f]
 */
inp::Grid build_group_velocity_grid(
    inp::Grid const& refractive_index_grid,
    NonuniformGridCalculator refractive_index_calculator)
{
    using namespace celeritas::literals;
    CELER_EXPECT(refractive_index_grid);

    // Construct the derivative of the refractive index with respect to energy
    // using derivative grid construction
    inp::Grid rindex_derivative
        = construct_derivative_grid(refractive_index_grid);

    // Calculate group velocity for each energy in the derivative grid
    inp::Grid result;
    result.x = rindex_derivative.x;
    result.y.resize(result.x.size());
    result.interpolation = rindex_derivative.interpolation;

    for (size_type i = 0; i < result.x.size(); ++i)
    {
        real_type const energy = result.x[i];
        real_type const rindex = refractive_index_calculator(energy);
        real_type const rindex_derivative_val = rindex_derivative.y[i];
        real_type const dispersion = energy * rindex_derivative_val;

        // Normal dispersion requires this term to be nonnegative, but tolerate
        // roundoff at zero.
        CELER_ASSERT(dispersion >= -std::numeric_limits<double>::epsilon());

        real_type const group_velocity
            = constants::c_light / celeritas::max(rindex + dispersion, 1_r);

        CELER_ASSERT(group_velocity <= constants::c_light / rindex);

        result.y[i] = group_velocity;
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
