//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadricSphereConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "corecel/math/SoftEqual.hh"

#include "../Plane.hh"
#include "../SimpleQuadric.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Try to convert a simple quadric to a sphere.
 *
 * The simple quadric must have already undergone normalization (zero or one
 * negative signs in the second order terms).
 */
class QuadricSphereConverter
{
  public:
    // Construct with tolerance
    inline QuadricSphereConverter(real_type tol);

    // Try converting to a sphere
    std::optional<Sphere> operator()(SimpleQuadric const& sq) const;

  private:
    SoftEqual<> soft_equal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance.
 */
QuadricSphereConverter::QuadricSphereConverter(real_type tol)
    : soft_equal_{tol}
{
}

//---------------------------------------------------------------------------//
/*!
 * Try converting to a sphere with this orientation.
 *
 * \verbatim
   x^2 + y^2 + z^2
       - 2x_0 - 2y_0 - 2z_0
       + x_0^2 + y_0^2 + z_0^2 - r^2 = 0
 * \endverbatim
 * SQ:
 * \verbatim
   a x^2 + a y^2 + a z^2 + e x + f y + g z  + h = 0
 * \endverbatim
 * so
 * \verbatim
    -2 x_0 = e/b --> x_0 = e / (-2 * a)
    -2 y_0 = f/b --> y_0 = f / (-2 * a)
    -2 z_0 = g/b --> z_0 = g / (-2 * a)
   x_0^2 + y_0^2 + z_0^2 - r^2 = h/b --> r^2 = (origin . origin - h/b)
 * \endverbatim
 *
 * All the coefficients should be positive or nearly so.
 */
std::optional<Sphere>
QuadricSphereConverter::operator()(SimpleQuadric const& sq) const
{
    CELER_EXPECT(std::all_of(
        sq.second().begin(), sq.second().end(), [this](real_type v) {
            return v >= 0 || soft_equal_(v, 0);
        }));
    constexpr auto X = to_int(Axis::x);
    constexpr auto Y = to_int(Axis::y);
    constexpr auto Z = to_int(Axis::z);

    auto second = sq.second();
    if (!soft_equal_(second[X], second[Y])
        || !soft_equal_(second[X], second[Z]))
    {
        // Coefficients aren't equal: it's some sort of ellipsoid
        return {};
    }

    real_type const inv_norm = 3 / (second[0] + second[1] + second[2]);
    CELER_ASSERT(inv_norm > 0);

    Real3 origin = make_array(sq.first());
    origin *= real_type{-0.5} * inv_norm;

    real_type radius_sq = dot_product(origin, origin) - sq.zeroth() * inv_norm;
    if (radius_sq <= 0)
    {
        // No real solution
        return {};
    }

    // Clear potential signed zeros before returning
    origin += real_type{0};
    return Sphere::from_radius_sq(origin, radius_sq);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
