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
 * Promote implicitly from a centered sphere.
 */
Sphere::Sphere(SphereCentered const& other)
    : origin_{0, 0, 0}, radius_sq_{other.radius_sq()}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
