//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Plane.cc
//---------------------------------------------------------------------------//
#include "Plane.hh"

#include "PlaneAligned.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Promote from an axis-aligned plane.
 */
template<Axis T>
Plane::Plane(PlaneAligned<T> const& other) noexcept
    : normal_{other.calc_normal()}, d_{other.position()}
{
}

//---------------------------------------------------------------------------//

//! \cond
template Plane::Plane(PlaneAligned<Axis::x> const&) noexcept;
template Plane::Plane(PlaneAligned<Axis::y> const&) noexcept;
template Plane::Plane(PlaneAligned<Axis::z> const&) noexcept;
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace celeritas
