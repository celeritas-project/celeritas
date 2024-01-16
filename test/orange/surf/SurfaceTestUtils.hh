//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/cont/Array.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Calculate intersections, sorting so that "no intersection" is at end.
 */
template<class S>
typename S::Intersections calc_intersections(S const& surf,
                                             Real3 const& pos,
                                             Real3 const& dir,
                                             SurfaceState on_surface)
{
    auto result = surf.calc_intersections(pos, dir, on_surface);
    auto ignored = [](real_type v) { return v == no_intersection() || v <= 0; };

    std::sort(result.begin(),
              result.end(),
              [&ignored](real_type left, real_type right) {
                  if (ignored(left) && !ignored(right))
                      return false;
                  if (!ignored(left) && ignored(right))
                      return true;
                  return left < right;
              });
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
