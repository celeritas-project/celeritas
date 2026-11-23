//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cmath>

#include "corecel/Constants.hh"
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
typename S::Intersections calc_intersections(
    S const& surf, Real3 const& pos, Real3 const& dir, SurfaceState on_surface)
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
/*!
 * Estimate the distance to a surface by sampling straight-line intersections.
 *
 * Directions are spread uniformly over the sphere with a Fibonacci lattice.
 * The nearest intersection over all directions is never smaller than the true
 * distance and converges to it as the number of directions increases, so a
 * valid safety must never exceed it.
 */
template<class S>
real_type calc_sampled_distance(
    S const& surf, Real3 const& pos, int num_dirs = 20000)
{
    real_type const golden_angle = constants::pi
                                   * (3 - std::sqrt(real_type(5)));
    real_type result = no_intersection();
    for (int i = 0; i < num_dirs; ++i)
    {
        real_type const z = 1 - (2 * real_type(i) + 1) / num_dirs;
        real_type const r = std::sqrt(1 - z * z);
        real_type const phi = golden_angle * i;
        Real3 const dir{r * std::cos(phi), r * std::sin(phi), z};
        for (real_type d : surf.calc_intersections(pos, dir, SurfaceState::off))
        {
            if (d > 0)
            {
                result = std::min(result, d);
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
