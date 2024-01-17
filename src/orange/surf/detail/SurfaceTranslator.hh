//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceTranslator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"
#include "orange/transform/Translation.hh"

#include "../SurfaceFwd.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply a translation to a surface to get another surface.
 *
 * The translation is the new origin for the surface, i.e. daughter-to-parent.
 * A sphere centered about the origin will be translated to a sphere with its
 * center on the given point.
 */
class SurfaceTranslator
{
  public:
    //! Construct with the new origin of the surface
    explicit SurfaceTranslator(Translation const& trans) : tr_{trans} {}

    //// SURFACE FUNCTIONS ////

    template<Axis T>
    PlaneAligned<T> operator()(PlaneAligned<T> const&) const;

    template<Axis T>
    CylAligned<T> operator()(CylCentered<T> const&) const;

    Sphere operator()(SphereCentered const&) const;

    template<Axis T>
    CylAligned<T> operator()(CylAligned<T> const&) const;

    Plane operator()(Plane const&) const;

    Sphere operator()(Sphere const&) const;

    template<Axis T>
    ConeAligned<T> operator()(ConeAligned<T> const&) const;

    SimpleQuadric operator()(SimpleQuadric const&) const;

    GeneralQuadric operator()(GeneralQuadric const&) const;

  private:
    Translation tr_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
