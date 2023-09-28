//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadricCylConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "corecel/math/SoftEqual.hh"
#include "orange/surf/CylAligned.hh"
#include "orange/surf/SimpleQuadric.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Try to convert a simple quadric to a cylinder.
 *
 * The simple quadric must have already undergone normalization (one
 * second-order term approximately zero, the others positive).
 */
class QuadricCylConverter
{
  public:
    // Construct with tolerance
    inline QuadricCylConverter(real_type tol);

    // Try converting to a cylinder with this orientation
    template<Axis T>
    std::optional<CylAligned<T>>
    operator()(AxisTag<T>, SimpleQuadric const& sq) const;

  private:
    SoftEqual<> soft_equal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance.
 */
QuadricCylConverter::QuadricCylConverter(real_type tol) : soft_equal_{tol} {}

//---------------------------------------------------------------------------//
/*!
 * Try converting to a cylinder with this orientation.
 *
 * Cone:
 * \verbatim
    (x - x_0)^2 + (y - y_0)^2 - r^2 = 0
 * \endverbatim
 * Expanded:
 * \verbatim
           x^2 +      y^2
    - 2x_0 x   - 2y_0 y
    + (x_0^2 + y_0^2 - r^2) = 0
 * \endverbatim
 * SQ:
 * \verbatim
   a x^2 + (a + \epsilon) y^2 + e x + f y + h = 0
 * \endverbatim
 * Normalized (\c epsilon = 0)
 * \verbatim
   x^2 + y^2 + e/b x + f/b y + g/b z  + h/b = 0
 * \endverbatim
 * Match terms:
 * \verbatim
    -2 x_0 = e/b --> x_0 = e / (-2 * b)
    -2 y_0 = f/b --> y_0 = f / (-2 * b)
   (x_0^2 + y_0^2 - r^2) = h/b
      -> r^2 = x_0^2 + y_0^2 - h/b
 * \endverbatim
 */
template<Axis T>
std::optional<CylAligned<T>>
QuadricCylConverter::operator()(AxisTag<T>, SimpleQuadric const& sq) const
{
    // Other coordinate system
    constexpr auto U = CylAligned<T>::u_axis();
    constexpr auto V = CylAligned<T>::v_axis();

    auto second = sq.second();
    if (!soft_equal_(0, second[to_int(T)]))
    {
        // Not the zero component we're looking for
        return {};
    }
    if (!soft_equal_(0, sq.first()[to_int(T)]))
    {
        // Not an axis-aligned cylinder
        return {};
    }
    if (!soft_equal_(second[to_int(U)], second[to_int(V)]))
    {
        // Not a *circular* cylinder
        return {};
    }
    CELER_ASSERT(second[to_int(U)] > 0 && second[to_int(V)] > 0);

    // Normalize so U, V second-order coefficients are 1
    auto const inv_norm = 2 / (second[to_int(U)] + second[to_int(V)]);

    // Calculate origin from first-order coefficients
    Real3 origin{0, 0, 0};
    origin[to_int(U)] = real_type{-0.5} * inv_norm * sq.first()[to_int(U)];
    origin[to_int(V)] = real_type{-0.5} * inv_norm * sq.first()[to_int(V)];

    real_type radius_sq = ipow<2>(origin[to_int(U)])
                          + ipow<2>(origin[to_int(V)]) - sq.zeroth() * inv_norm;

    if (radius_sq <= 0)
    {
        // No real solution
        return {};
    }
    return CylAligned<T>::from_radius_sq(origin, radius_sq);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
