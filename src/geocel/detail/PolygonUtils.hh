//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/LengthUnits.hh
//! \brief NOTE: only use inside geocel; prefer celeritas/Units.hh
//---------------------------------------------------------------------------//
#pragma once

//#define CELER_ICRT inline constexpr real_type
#include <iostream>
#include <vector>
#include <corecel/cont/Array.hh>

namespace celeritas
{
namespace geoutils
{

template<typename T>
using Real2 = Array<T, 2>;
template<typename T>
using VecReal2 = std::vector<Real2<T>>;

//---------------------------------------------------------------------------//
/*! Check if a 2D polygon is convex.
 *
 * /param corners the vertices of the polygon
 * /param degen_ok if true, consecutive degenerate points are okay, still
 * return true
 */
template<typename T>
CELER_FUNCTION bool
IsPolygonConvex(VecReal2<T> const& corners, bool degen_ok = false)
{
    // The cross product of all vector pairs corresponding to ordered
    // consecutive segments has to be positive.
    auto crossp = [&](Real2<T> const& v1, Real2<T> const& v2) {
        return v1[0] * v2[1] - v1[1] * v2[0];
    };

    auto vecsub = [&](Real2<T> const& v1, Real2<T> const& v2) {
        return Real2<T>{v1[0] - v2[0], v1[1] - v2[1]};
    };

    // use the cross_product(last, first) as sign reference
    auto N = corners.size();
    auto vec1 = vecsub(corners[0], corners[N - 1]);
    auto vec2 = vecsub(corners[1], corners[0]);
    auto ref = crossp(vec1, vec2);
    for (size_type i = 0; i < N - 1; ++i)
    {
        vec1 = vec2;
        vec2 = vecsub(corners[(i + 2) % N], corners[(i + 1) % N]);
        T val = crossp(vec1, vec2);
        // make sure the sign is the same as the reference
        if (val * ref < T{0} || (!degen_ok && val == T{0}))
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace geoutils
}  // namespace celeritas
