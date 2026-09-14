//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/GroupVelocityGridBuilder.cc
//---------------------------------------------------------------------------//
#include "GroupVelocityGridBuilder.hh"

#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
inp::Grid optical::GroupVelocityGridBuilder::operator()(
    inp::Grid const& ri) const
{
    CELER_EXPECT(ri);

    // Construct the derivative of the refractive index with respect to energy
    // using derivative grid construction
    inp::Grid rindex_derivative = construct_derivative_grid(ri);

    // Calculate group velocity for each energy in the derivative grid
    inp::Grid result;
    result.x = rindex_derivative.x;
    result.y.resize(result.x.size());
    result.interpolation = rindex_derivative.interpolation;

    for (size_type i = 0; i < result.x.size(); ++i)
    {
        real_type const energy = result.x[i];
        real_type const rindex = refractive_index_(energy);
        real_type const rindex_derivative_val = rindex_derivative.y[i];
        real_type const group_velocity
            = constants::c_light / (rindex + energy * rindex_derivative_val);

        real_type const phase_velocity = constants::c_light / rindex;

        result.y[i] = (group_velocity > 0 && group_velocity <= phase_velocity)
                          ? group_velocity
                          : phase_velocity;
    }

    CELER_ENSURE(result);
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
