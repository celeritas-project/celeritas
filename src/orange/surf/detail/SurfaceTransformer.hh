//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceTransformer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"
#include "orange/transform/Transformation.hh"

#include "../SurfaceFwd.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply a transformation to a surface to get another surface.
 *
 * The transform gives the new origin and rotation for the surface: rotation is
 * applied first, then translation.
 */
class SurfaceTransformer
{
  public:
    //! Construct with the transformation to apply
    explicit SurfaceTransformer(Transformation const& trans) : tr_{trans} {}

    //// SURFACE FUNCTIONS ////

    template<Axis T>
    Plane operator()(PlaneAligned<T> const&) const;

    template<Axis T>
    GeneralQuadric operator()(CylCentered<T> const&) const;

    Sphere operator()(SphereCentered const&) const;

    template<Axis T>
    GeneralQuadric operator()(CylAligned<T> const&) const;

    Plane operator()(Plane const&) const;

    Sphere operator()(Sphere const&) const;

    template<Axis T>
    GeneralQuadric operator()(ConeAligned<T> const&) const;

    GeneralQuadric operator()(SimpleQuadric const&) const;

    GeneralQuadric operator()(GeneralQuadric const&) const;

  private:
    Transformation tr_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
