//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/BumpPosition.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bump each component of a position by a distance toward a direction.
 *
 * Each component with a nonzero direction moves by the full distance toward
 * the sign of that direction component, regardless of its magnitude. Unlike
 * a linear move along the direction, this moves the position off a surface
 * that the direction is nearly tangent to, and it moves a component near
 * zero by the same absolute distance as a large one. The displacement is not
 * parallel to the direction, but it is always in the same octant and at most
 * \f$ \sqrt{3} \f$ times the distance. A zero direction component is not
 * moved.
 */
inline CELER_FUNCTION void bump_position(
    real_type dist, Real3 const& dir, Real3* pos)
{
    CELER_EXPECT(dist >= 0);
    CELER_EXPECT(pos);
    for (auto i : {0, 1, 2})
    {
        if (dir[i] != 0)
        {
            (*pos)[i] += std::copysign(dist, dir[i]);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
