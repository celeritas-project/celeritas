//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate exiting direction via conservation of momentum.
 */
inline CELER_FUNCTION Real3 calc_exiting_direction(real_type inc_momentum,
                                                   real_type out_momentum,
                                                   Real3 const& inc_direction,
                                                   Real3 const& out_direction)
{
    Real3 result;
    for (int i = 0; i < 3; ++i)
    {
        result[i] = inc_direction[i] * inc_momentum
                    - out_direction[i] * out_momentum;
    }
    return make_unit_vector(result);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
