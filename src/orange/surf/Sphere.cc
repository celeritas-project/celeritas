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
 * Promote from a centered sphere.
 */
Sphere::Sphere(SphereCentered const& other)
    : origin_{0, 0, 0}, radius_sq_{other.radius_sq()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a new origin and the radius of another sphere.
 */
Sphere::Sphere(Real3 const& origin, Sphere const& other)
    : origin_{origin}, radius_sq_{other.radius_sq_}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
