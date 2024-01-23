//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadricConeConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "corecel/math/SoftEqual.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/SimpleQuadric.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Try to convert a simple quadric to a cone.
 *
 * The simple quadric must have already undergone normalization (one
 * second-order term less than zero, two positive).
 */
class QuadricConeConverter
{
  public:
    // Construct with tolerance
    inline QuadricConeConverter(real_type tol);

    // Try converting to a cone with this orientation
    template<Axis T>
    std::optional<ConeAligned<T>>
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
QuadricConeConverter::QuadricConeConverter(real_type tol) : soft_equal_{tol} {}

//---------------------------------------------------------------------------//
/*!
 * Try converting to a cone with this orientation.
 *
 * Cone:
 * \verbatim
    -t^2 (x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2 = 0
 * \endverbatim
 * Expanded:
 * \verbatim
           -t^2 x^2    + y^2    + z^2
    + 2 t^2 x_0 x - 2y_0 y - 2z_0 z
    + (- t^2 x_0^2 + y_0^2 + z_0^2) = 0
 * \endverbatim

 * SQ:
 * \verbatim
   a x^2 + b y^2 + (b + \epsilon) z^2 + e x + f y + g z  + h = 0
 * \endverbatim
 * Normalized (\c epsilon = 0)
 * \verbatim
   a/b x^2 + y^2 + z^2 + e/b x + f/b y + g/b z  + h/b = 0
 * \endverbatim
 * Match terms:
 * \verbatim
        -t^2  = a/b
    2 t^2 x_0 = e/b --> x_0 = e / (2 * a)
       -2 y_0 = f/b --> y_0 = f / (-2 * b)
       -2 z_0 = g/b --> z_0 = g / (-2 * b)
   - t^2 x_0^2 + y_0^2 + z_0^2 = h/b    [if not, it's a hyperboloid!]
 * \endverbatim
 */
template<Axis T>
std::optional<ConeAligned<T>>
QuadricConeConverter::operator()(AxisTag<T>, SimpleQuadric const& sq) const
{
    // Other coordinate system
    constexpr auto U = ConeAligned<T>::u_axis();
    constexpr auto V = ConeAligned<T>::v_axis();

    auto second = sq.second();
    if (!(second[to_int(T)] < 0))
    {
        // Not the negative component
        return {};
    }
    if (!soft_equal_(second[to_int(U)], second[to_int(V)]))
    {
        // Not a circular cone
        return {};
    }
    CELER_ASSERT(second[to_int(U)] > 0 && second[to_int(V)] > 0);

    // Normalize so U, V second-order coefficients are 1
    real_type const norm = (second[to_int(U)] + second[to_int(V)]) / 2;
    real_type const tsq = -second[to_int(T)] / norm;

    // Calculate origin from first-order coefficients
    Real3 origin = make_array(sq.first());
    origin[to_int(T)] /= -2 * second[to_int(T)];
    origin[to_int(U)] /= -2 * norm;
    origin[to_int(V)] /= -2 * norm;

    real_type const expected_h_b = -tsq * ipow<2>(origin[to_int(T)])
                                   + ipow<2>(origin[to_int(U)])
                                   + ipow<2>(origin[to_int(V)]);
    if (!soft_equal_(expected_h_b, sq.zeroth() / norm))
    {
        // Leftover constant: it's a hyperboloid!
        return {};
    }

    return ConeAligned<T>::from_tangent_sq(origin, tsq);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
