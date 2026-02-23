//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.cc
//---------------------------------------------------------------------------//
#include "Toroid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct toroid from origin point and radii.
 *
 * \param origin 3d origin of the toroid.
 * \param major_radius Radius from origin to the center of revolved ellipse.
 * \param ellipse_xy_radius Radius of ellipse in xy plane, aka 'a'.
 * \param ellipse_z_radius Radius of ellipse aligned with z axis, aka 'b'.
 */
Toroid::Toroid(Real3 const& origin,
               real_type major_radius,
               real_type ellipse_xy_radius,
               real_type ellipse_z_radius)
    : origin_(origin)
    , r_(major_radius)
    , a_(ellipse_xy_radius)
    , b_(ellipse_z_radius)
{
    CELER_EXPECT(r_ > 0);
    CELER_EXPECT(a_ > 0);
    CELER_EXPECT(b_ > 0);
    CELER_EXPECT(r_ > a_);  // Degenerate torii
}
}  // namespace celeritas
