//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/BumpUlp.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/NumericLimits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bump each component of a position by a number of ULPs along a direction.
 *
 * Each component with a nonzero direction moves to the \c n -th next
 * representable value toward the sign of that direction component, regardless
 * of the magnitude of the direction. This guarantees that the position
 * changes in every direction it is moving, even when a linear push along a
 * nearly tangent direction is lost to roundoff. The displacement is not
 * parallel to the direction, but it is always in the same octant.
 */
inline CELER_FUNCTION void bump_ulp(
    unsigned int n, Real3 const& dir, Real3* pos)
{
    CELER_EXPECT(pos);
    constexpr auto inf = numeric_limits<real_type>::infinity();
    for (auto i : {0, 1, 2})
    {
        if (dir[i] == 0)
        {
            continue;
        }
        real_type const target = dir[i] > 0 ? inf : -inf;
        real_type& p = (*pos)[i];
        for (unsigned int j = 0; j < n; ++j)
        {
            p = std::nextafter(p, target);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
