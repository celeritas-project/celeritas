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
 * Bump each component of a position by about \c n ULPs along a direction.
 *
 * Each component with a nonzero direction moves by \c n machine epsilons
 * relative to its own magnitude (between \c n and \c 2n ULP) toward the sign
 * of that direction component, regardless of the magnitude of the direction.
 * This guarantees that the position changes in every direction it is moving,
 * even when a linear push along a nearly tangent direction is lost to
 * roundoff. The displacement is not parallel to the direction, but it is
 * always in the same octant. A zero component is not bumped, since a linear
 * push near zero is not truncated.
 */
inline CELER_FUNCTION void bump_ulp(
    unsigned int n, Real3 const& dir, Real3* pos)
{
    CELER_EXPECT(pos);
    real_type const rel = n * numeric_limits<real_type>::epsilon();
    for (auto i : {0, 1, 2})
    {
        if (dir[i] == 0)
        {
            continue;
        }
        real_type& p = (*pos)[i];
        p = std::fma(std::copysign(rel, dir[i]), std::fabs(p), p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
