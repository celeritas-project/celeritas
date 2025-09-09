//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

#include "SurfaceFwd.hh"
#include "VariantSurface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Truncate a bounding zone using a convex quadric surface.
 *
 * Convex surfaces are planes, spheroids, cylinders, parabolic cylinders, and
 * paraboloids. All but the spheroids are infinite in at least one direction.
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
 *
 * TODO: move to orangeinp/detail, use BZone, combine with
 * NegatedSurfaceClipper.
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

    void operator()(Involute const&) const;

    // Apply to a surface with unknown type
    void operator()(VariantSurface const& surf) const;

  private:
    BBox* int_{nullptr};
    BBox* ext_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
