//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

#include "SurfaceFwd.hh"
#include "VariantSurface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Truncate a bounding zone using a surface.
 *
 * This \em reduces the size of inner and outer bounding boxes to fit a
 * surface. The \c interior bounding box is modified to be entirely \em inside
 * the surface, and the \c exterior is modified to be entirely \em outside.
 * Axes that cannot be determined inside or out are left unchanged.
 *
 * Even though most quadric surfaces are infinite, their intersection with a
 * bounding box may be a smaller bounding box. Accounting for the current
 * bounding box's size when considering further truncation is *not yet
 * implemented*.
 *
 * Shrinking bounding boxes will accelerate "distance to in" and "distance
 * to out" calculations.
 */
class SurfaceClipper
{
  public:
    // Construct with interior and exterior bounding boxes
    explicit SurfaceClipper(BBox* interior, BBox* exterior);

    //// OPERATION ////

    // Apply to a surface with a known type
    template<Axis T>
    void operator()(PlaneAligned<T> const&) const;

    template<Axis T>
    void operator()(CylCentered<T> const&) const;

    void operator()(SphereCentered const&) const;

    template<Axis T>
    void operator()(CylAligned<T> const&) const;

    void operator()(Plane const&) const;

    void operator()(Sphere const&) const;

    template<Axis T>
    void operator()(ConeAligned<T> const&) const;

    void operator()(SimpleQuadric const&) const;

    void operator()(GeneralQuadric const&) const;

    // Apply to a surface with unknown type
    void operator()(VariantSurface const& surf) const;

  private:
    BBox* int_;
    BBox* ext_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
