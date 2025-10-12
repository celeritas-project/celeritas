//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadricPlaneConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/SoftEqual.hh"

#include "../Plane.hh"
#include "../SimpleQuadric.hh"

namespace celeritas
{
namespace detail
{
//-------------------------------------------------------------------------//
/*!
 * Convert a simple quadric to a plane.
 */
class QuadricPlaneConverter
{
  public:
    // Construct with tolerance
    inline QuadricPlaneConverter(real_type tol);

    // Convert to a plane
    Plane operator()(SimpleQuadric const& sq) const;

  private:
    SoftZero<> soft_zero_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance.
 */
QuadricPlaneConverter::QuadricPlaneConverter(real_type tol) : soft_zero_{tol}
{
}

//---------------------------------------------------------------------------//
/*!
 * Convert to a plane.
 *
 * Simple quadric is calculated as
 * dx + ey + fz + g = 0, but plane is
 * ax + bx + cz - d = 0
 * so we need to reverse the sign of the scalar component. We also need to
 * normalize with the same factor.
 */
Plane QuadricPlaneConverter::operator()(SimpleQuadric const& sq) const
{
    CELER_EXPECT(
        std::all_of(sq.second().begin(), sq.second().end(), soft_zero_));
    CELER_EXPECT(
        !std::all_of(sq.first().begin(), sq.first().end(), soft_zero_));
    // Second-order coefficients are zero: return a plane
    auto n = make_array(sq.first());

    real_type norm_factor = 1 / celeritas::norm(n);
    n *= norm_factor;

    real_type d = -sq.zeroth() * norm_factor;

    return Plane{n, d};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
