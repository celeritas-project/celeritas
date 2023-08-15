//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Sphere.cc
//---------------------------------------------------------------------------//
#include "Sphere.hh"

#include "SphereCentered.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from the square of the radius.
 *
 * This is used for surface simplification.
 */
Sphere Sphere::from_radius_sq(Real3 const& origin, real_type rsq)
{
    CELER_EXPECT(rsq > 0);
    Sphere result;
    result.origin_ = origin;
    result.radius_sq_ = rsq;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Promote from a centered sphere.
 */
Sphere::Sphere(SphereCentered const& other) noexcept
    : origin_{0, 0, 0}, radius_sq_{other.radius_sq()}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
